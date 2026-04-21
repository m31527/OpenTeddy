FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴（subprocess 沙盒可能需要）
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 建立 skills 目錄（若不存在）
RUN mkdir -p skills

# 暴露 port
EXPOSE 8000

# 啟動指令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
