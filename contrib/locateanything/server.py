#!/usr/bin/env python3
"""
A pointing service for OpenTeddy's browser_click_at, backed by NVIDIA's
LocateAnything-3B — OPTIONAL, and installed by you, not by OpenTeddy.

  ⚠ LocateAnything-3B is released under the NVIDIA License for
    NON-COMMERCIAL use: academic and non-profit RESEARCH only; commercial use
    is not permitted (https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE).
    OpenTeddy does not ship, download or enable it. Whether your use fits
    that license is yours to decide — see README "Optional: LocateAnything".

What it does: POST /point {"image": <base64 JPEG/PNG>, "query": "the heart
icon"} → {"point": [x, y]} on a 0–1000 scale (or {"point": null}); GET
/health. OpenTeddy calls it when Settings → "Pointing service" holds its
address; without it the regular vision model points.

Run it in its own Python environment (the model pins its own library
versions), on a machine with an NVIDIA GPU:

    python -m venv ~/la-venv && ~/la-venv/bin/pip install -r contrib/locateanything/requirements.txt
    ACCEPT_NVIDIA_NONCOMMERCIAL_LICENSE=yes ~/la-venv/bin/python contrib/locateanything/server.py --port 8090

The first run downloads the weights (~8 GB) from Hugging Face.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LICENSE_URL = "https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE"
POINT = re.compile(r"<box><(\d+)><(\d+)></box>")


def load(model_id: str):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda").eval()
    return tok, proc, model


def point(tok, proc, model, image, query: str):
    """The model card's "Point to: …" prompt; coordinates come back 0–1000."""
    import torch
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": f"Point to: {query}."}]}]
    text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(messages)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tok, max_new_tokens=256, use_cache=True, generation_mode="hybrid",
            temperature=0.7, do_sample=True, top_p=0.9, repetition_penalty=1.1, verbose=False)
    answer = out[0] if isinstance(out, tuple) else out
    m = POINT.search(str(answer))
    return [int(m.group(1)), int(m.group(2))] if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="LocateAnything pointing service for OpenTeddy (opt-in).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    args = ap.parse_args()

    if os.environ.get("ACCEPT_NVIDIA_NONCOMMERCIAL_LICENSE", "").lower() != "yes":
        print("LocateAnything-3B is licensed for non-commercial, academic / non-profit research use only.\n"
              f"Read {LICENSE_URL}, and if your use fits it, start again with\n"
              "  ACCEPT_NVIDIA_NONCOMMERCIAL_LICENSE=yes", file=sys.stderr)
        return 2

    from PIL import Image
    tok, proc, model = load(args.model)
    print(f"LocateAnything ready — POST http://{args.host}:{args.port}/point", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"ok": True, "model": args.model} if self.path == "/health" else {"error": "not found"},
                       200 if self.path == "/health" else 404)

        def do_POST(self):
            if self.path != "/point":
                return self._send({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                image = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
                self._send({"point": point(tok, proc, model, image, str(req["query"]))})
            except Exception as exc:  # noqa: BLE001
                self._send({"error": str(exc)}, 400)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
