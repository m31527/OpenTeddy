# OpenTeddy voice service (server.py): SenseVoice + Fun-CosyVoice3.
#
# Built on the vLLM image because on a DGX Spark (GB10, aarch64) its torch
# and torchaudio are built for the GPU — pip has no aarch64 CUDA wheels to
# mix in. Everything installed below is held to the image's own torch /
# torchaudio / numpy (constraints), so a dependency can't swap them out.
#
# transformers is NOT the image's: the vLLM base has 5.x, and CosyVoice's
# Qwen2 on it babbled — speech-shaped sound with no words in it (demo2,
# 11/20). CosyVoice is written against 4.51.3; diffusers 0.29 and
# huggingface_hub < 1.0 go with it. (This image doesn't run vLLM.)
#
# BASE: the aarch64 build for a DGX Spark; on an x86-64 machine with an
# NVIDIA GPU, vllm/vllm-openai:v0.33.0 (not tried yet). A machine whose
# local model already runs on vLLM has this image, so nothing is pulled.
ARG BASE=vllm/vllm-openai:v0.33.0-aarch64
FROM ${BASE}

ENTRYPOINT []
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 COSYVOICE_DIR=/opt/CosyVoice MODELS_DIR=/models VOICES_DIR=/voices
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs libsndfile1 && rm -rf /var/lib/apt/lists/*

ARG COSYVOICE_REF=main
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
 && cd /opt/CosyVoice && git checkout "$COSYVOICE_REF" && git submodule update --init --recursive

RUN pip freeze | grep -iE '^(torch|torchaudio|numpy|triton)==' > /tmp/keep.txt && cat /tmp/keep.txt
# CosyVoice's inference needs (its requirements.txt also lists training —
# deepspeed, pyworld —, the gradio demo and x86-only onnxruntime-gpu /
# tensorrt wheels; a probe of every module found only those missing).
RUN pip install -c /tmp/keep.txt \
      conformer==0.3.2 diffusers==0.29.0 transformers==4.51.3 "huggingface_hub<1.0" hydra-core==1.3.2 HyperPyYAML==1.2.3 inflect librosa \
      "modelscope>=1.20" omegaconf onnx onnxruntime pyarrow soundfile x-transformers wetext \
      fastapi uvicorn opencc-python-reimplemented lightning gdown wget matplotlib rich \
 && pip install -c /tmp/keep.txt --no-deps openai-whisper \
 && pip install -c /tmp/keep.txt tiktoken more-itertools numba
# SenseVoice through FunASR.
RUN pip install -c /tmp/keep.txt funasr

COPY server.py /opt/voice/server.py
WORKDIR /opt/voice
EXPOSE 8095
CMD ["python3", "/opt/voice/server.py"]
