# OpenTeddy's LiveKit agent (agent.py). Settings come from the environment
# (README.md): OPENTEDDY_URL / OPENTEDDY_KEY, LIVEKIT_URL / _API_KEY / _API_SECRET.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY agent.py .
# Silero's model ships inside the plugin; load it once so a broken install fails the build.
RUN OPENTEDDY_KEY=build python -c "from livekit.plugins import silero; silero.VAD.load()"
CMD ["python", "agent.py", "start"]
