<div align="center">

<sub><a href="README.md">English</a> | 繁體中文</sub>

<img src="static/OpenTeddy-logo.svg" alt="OpenTeddy" width="240" />

# OpenTeddy

**一個免費、類 Claude 的本機 AI 代理人**

本機模型 + 技能（Skills）+ 少量商業模型的混搭方案。

🌐 **官網：** [openteddy-72cee.web.app](https://openteddy-72cee.web.app/) &nbsp;·&nbsp; 📦 **原始碼：** [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

</div>

---

## 這是什麼？

OpenTeddy 的目標是做出一個**免費、可以在自己電腦上跑**的類 Claude 體驗，方法是把三層能力疊在一起：

1. **本機模型（Gemma / Qwen，透過 Ollama）** — 負責大部分的規劃與執行，不用花錢、資料也不出你的電腦。
2. **技能（Skills）** — 重複做過的事情會被自動寫成 Python 函式，之後同樣的任務就不用再呼叫 LLM。
3. **商業模型（Claude）** — 只有在本機模型撐不住的時候才會被叫進來（逾時、信心不足、硬錯誤等）。

結果是一個**效能接近前沿模型、花費卻非常低**的混搭方案。Usage 頁面會用 GPT-4 API 的價格換算你目前為止省了多少錢，越用越有感。

> **如果你喜歡這個方向、或想給我一點鼓勵，請到 GitHub 幫忙點一顆 ⭐️，這對我來說是最大的動力！**
> → [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

## 主要特色

- **本機優先** — 規劃與執行都跑在你的機器上，Claude 只在必要時介入。
- **自動升級到 Claude** — 逾時、信心低、連續失敗、工具錯誤訊號都會自動觸發。
- **技能會自己長出來** — 重複的任務會被升級成可重用的 Python 技能，越用越快。
- **Web 儀表板** — 提交任務、即時看工具呼叫、審核敏感指令、管理記憶。
- **人類確認關卡** — `rm`、`sudo`、`mv` 等高風險指令會停下來等你同意。
- **長期記憶** — ChromaDB 記住過去的脈絡，下次規劃時自動帶進來。

## 快速開始（原生安裝，推薦）

### 1. 事前準備

- Python 3.11+
- [Ollama](https://ollama.ai)：
  ```bash
  ollama pull gemma3:4b
  ollama pull qwen2.5:3b
  ```
- 一把 Anthropic API key（用於升級與產生技能）

### 2. 安裝

```bash
git clone https://github.com/m31527/OpenTeddy.git
cd OpenTeddy
python -m venv .venv
source .venv/bin/activate   # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 設定

```bash
cp .env.example .env
# 編輯 .env，至少填入 ANTHROPIC_API_KEY
```

### 4. 啟動

```bash
uvicorn main:app --reload
# 儀表板： http://localhost:8000
# API 文件： http://localhost:8000/docs
```

## 平台支援

| 作業系統 | 狀態 | 備註 |
|----------|------|------|
| **macOS**（Intel / Apple Silicon） | ✅ 完全支援 | 主要開發環境 |
| **Linux** | ✅ 完全支援 | 任何有 Python 3.11+ 與 Ollama 的發行版 |
| **Windows（原生）** | ⚠️ 部分支援，建議改用 WSL2 | 見下方注意事項 |
| **Windows（WSL2）** | ✅ 完全支援 | 等同於 Linux，Windows 使用者推薦此路徑 |

### Windows 原生環境的注意事項

程式本身是跨平台 Python（用 `pathlib`、`os.path.join`、`asyncio`），但實際會踩到坑的地方是：

- **執行器 LLM 會產生 POSIX shell 指令。** 當 Qwen 決定跑 `ls`、`rm -rf`、`grep`、`chmod` 或 `cmd1 | tee file` 這類 pipe 指令時，會被交給系統 shell；在原生 Windows 上那是 **cmd.exe / PowerShell**，這些指令會失敗。裝到 **WSL2** 裡跑就沒這問題。
- **Windows 沒有 `lsof` / `ps`**，所以 `tools/deploy_tool.py` 裡跟埠口有關的工具會降級：`port_probe` 只能回報佔用與否、沒有 PID；`port_free` 無法用 port 殺行程。
- **Ollama 在 Windows** 是官方支援的（到 ollama.com 下載安裝），拉模型、跑模型都跟 Mac/Linux 一樣。

**建議做法：** Windows 使用者把 Ollama 裝在主機原生環境（可以吃 GPU），OpenTeddy 本體裝進 **WSL2 Ubuntu**。這樣同時拿到 GPU 加速與 POSIX 環境。

### Docker 網路的平台差異（Linux 主機）

`docker-compose.yml` 用了 `extra_hosts: ["host-gateway:host-gateway"]` 讓容器能連到主機上的 Ollama。在 Linux 上需要 **Docker Engine 20.10+**，而且 Ollama 必須綁在 `0.0.0.0`（不能只綁 `127.0.0.1`），容器走 bridge network 才連得到。啟動前設 `OLLAMA_HOST=0.0.0.0:11434` 再 `ollama serve`。Mac / Windows 的 Docker Desktop 則是直接可以用。

## Docker 部署

```bash
cp .env.example .env
# 填入 ANTHROPIC_API_KEY
docker compose up -d
# 開啟 http://localhost:8000
```

### ⚠️ Docker 版本看不到你電腦上的檔案

預設的 `docker-compose.yml` 只掛了一顆隔離的 named volume（`openteddy_data` → `/app/data`），**沒有**把你的家目錄、桌面、Downloads 或其他主機資料夾掛進容器。也就是說：

- 像是「幫我讀 `~/Documents/report.pdf`」、「整理我的 Downloads 資料夾」、「跑我桌面上這支腳本」這種任務，**在 Docker 版本裡是做不到的** — 容器根本看不到那些檔案。
- 代理人的 shell / file / python 工具完全在容器**內部**運作，讀寫的檔案都在 `/app/data` 裡，volume 被刪掉就不見了。

**如果你希望代理人直接操作你電腦上的檔案，請用原生 `uvicorn` 方式啟動**（見上方〈快速開始〉），不要用 Docker。原生行程擁有你使用者權限內的完整檔案系統存取權，這也是大多數「本機助理」情境真正需要的模式。

如果你真的想用 Docker，可以自己在 `docker-compose.yml` 加一個 bind mount，例如：

```yaml
    volumes:
      - openteddy_data:/app/data
      - ${HOME}/openteddy-workspace:/workspace   # ← 暴露給容器的主機資料夾
```

然後讓代理人只在容器內的 `/workspace` 裡面活動。只有你明確掛進去的資料夾會被看到，其餘仍然隔離。

## 更完整的文件

更多架構圖、API 列表、環境變數說明、自我成長機制與 Claude 介入條件表，請看 **[英文版 README](README.md)**，本檔只是精簡入門。

## 支持這個專案

OpenTeddy 是一個人在做的 side project，想證明一套小而開源的組合也能逼近商業代理人的體驗。如果你想支持它繼續長大：

- ⭐ **去 repo 點星** — [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy) — 對我是最大鼓勵。
- 🐛 **開 issue** — 遇到 bug、模型設定卡關都歡迎回報。
- 🧠 **分享你做的技能** — 歡迎 PR。

## 授權

MIT
