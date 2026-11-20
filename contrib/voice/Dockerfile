# OpenTeddy voice service (server.py): SenseVoice + Fun-CosyVoice3.
#
# Built on the vLLM image because on a DGX Spark (GB10, aarch64) its torch,
# torchaudio and transformers are a matched set built for the GPU — pip has
# no aarch64 CUDA wheels to mix in. Everything installed below is pinned to
# the image's own torch / torchaudio / transformers / numpy (constraints), so
# a dependency can't swap them out.
FROM vllm/vllm-openai:v0.33.0-aarch64

ENTRYPOINT []
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 COSYVOICE_DIR=/opt/CosyVoice MODELS_DIR=/models VOICES_DIR=/voices
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs libsndfile1 && rm -rf /var/lib/apt/lists/*

ARG COSYVOICE_REF=main
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
 && cd /opt/CosyVoice && git checkout "$COSYVOICE_REF" && git submodule update --init --recursive

RUN pip freeze | grep -iE '^(torch|torchaudio|transformers|numpy|triton|tokenizers)==' > /tmp/keep.txt && cat /tmp/keep.txt
# CosyVoice's inference needs (its requirements.txt also lists training —
# deepspeed, pyworld —, the gradio demo and x86-only onnxruntime-gpu /
# tensorrt wheels; a probe of every module found only those missing).
RUN pip install -c /tmp/keep.txt \
      conformer==0.3.2 "diffusers>=0.29" hydra-core==1.3.2 HyperPyYAML==1.2.3 inflect librosa \
      "modelscope>=1.20" omegaconf onnx onnxruntime pyarrow soundfile x-transformers wetext \
      huggingface_hub fastapi uvicorn opencc-python-reimplemented lightning gdown wget matplotlib rich \
 && pip install -c /tmp/keep.txt --no-deps openai-whisper \
 && pip install -c /tmp/keep.txt tiktoken more-itertools numba
# SenseVoice through FunASR.
RUN pip install -c /tmp/keep.txt funasr

COPY server.py /opt/voice/server.py
WORKDIR /opt/voice
EXPOSE 8095
CMD ["python3", "/opt/voice/server.py"]
