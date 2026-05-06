<div align="center">

<sub><a href="README.md">English</a> | 繁體中文</sub>

<img src="static/OpenTeddy-logo.svg" alt="OpenTeddy" width="240" />

# OpenTeddy

**讓本機 LLM 真正能交付工作的平台。**

本機模型自己跑不太動。OpenTeddy 在它外面包一層 — 強化過的 agent loop、會自動長出來的技能庫，以及恰到好處的商業模型補位 — 把它變成真的能完成任務的代理人。

🌐 **官網：** [openteddy.net](https://openteddy.net/) &nbsp;·&nbsp; 📦 **原始碼：** [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

</div>

---

## 為什麼存在這個專案

2B / 4B / 7B 級的本機模型，自己單打獨鬥就是個玩具：會幻覺、會跳針、做到一半放棄。**模型本身不是產品，外面那層平台才是**。OpenTeddy 就是那一層：

- **強化過的 agent loop** — 知道何時該放棄、何時該重試、何時該叫 Claude，不會無限「讓我再試一次」鬼打牆。
- **自我成長的技能庫** — 重複的工作會被寫成 Python 函式，下次再問同樣的事情完全不用呼叫 LLM。
- **依硬體分級的模型預設組合** — 從 16 GB MacBook 到 DGX Spark，每個 tier 都有調好的 `num_ctx` / `max_tokens` / timeout。
- **商業模型作為安全網而非帳單** — Claude 只在地端真的卡住才被叫，Usage 頁面會告訴你 GPT-4 跑同樣工作要花多少。

結果：你的 $0 token 本機硬體真的把工作做完，sidebar 裡那個累積的 savings 數字，是讓你不再擔心 Claude Pro 自動續訂的真正原因。

> **如果你喜歡這個方向、或想給我一點鼓勵，請到 GitHub 幫忙點一顆 ⭐️，這對我來說是最大的動力！**
> → [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

## 主要特色

- **本機優先** — 規劃（Gemma）與執行（Qwen）都跑在你的機器上；Claude 只在地端撐不住時才被叫進來。
- **自動升級到 Claude** — 逾時、信心低、連續失敗、deliverable verifier 判 FAIL、工具錯誤訊號都會自動觸發。可用 `ESCALATION_ENABLED=false` 整個關掉。
- **Token 串流 UI** — orchestrator 規劃 + executor 回答都透過 WebSocket token-by-token 流出來，不再呆呆看著 spinner。
- **逐步交付物驗證** — 每個成功 subtask 結束會用 LLM-as-judge 確認產出的檔案真的是 deliverable（不是「描述 deliverable 的報告」），可在大模型上關掉。
- **小模型 loop 強化** — 自適應 prompt、低風險工具並行、重複呼叫上限、circuit breaker、發現 memo、context watchdog 自動壓縮舊輪次、釘住 session workspace 防路徑 drift。
- **永跑指令防呆** — shell tool 自動拒絕 `tail -f`、`journalctl -f`、`watch …`，並把 `docker compose up` 自動加上 `-d`，restart-crash 容器的 log 串流不會把 subtask 卡死。
- **內建 web search** — Chat mode 開啟 `web_search` 工具（Brave Search API），地端模型可以查最新資料而不是亂編訓練 cutoff 後的事件。
- **斷線可恢復的 WS** — 600 筆 ring buffer + `?since=` 重播，網路抖一下、刷新分頁都不會卡 UI。
- **技能會自己長出來** — 重複的任務會被升級成可重用的 Python 技能，越用越快。
- **Web 儀表板** — 提交任務、即時看工具呼叫、審核敏感指令、管理記憶、即時顯示「已省 $X vs GPT-4」，還有 GFM 表格 + Chart.js 數值標籤 HTML 報表。
- **Capabilities tab** — 內建 Tools 跟自動長出來的 Skills 合併在同一個可篩選列表，type badge 標明來源；技能用越多自動從 TESTING 升 ACTIVE。
- **macOS 原生客戶端** — Tauri 2.x 殼，引導精靈（Ollama 一鍵安裝 + 機器分級拉模型）、語言切換器、模式鎖定、可自由拖動的視窗、自動更新、診斷下載。詳見 [`desktop/`](desktop/)。
- **可選的雲端帳號** — 用 Google 登入即可跨裝置同步技能、記憶、設定。匿名 Firebase Auth 每個 install 從第一次開機就有；登入是 opt-in。OSS web UI 完全不含 Firebase code，雲端只在 desktop 殼裡跑。
- **Lemon Squeezy 終身授權** — 一次性 $99 解鎖簽名版桌面、雲端同步、未來的 premium 技能包。Open-source 核心永遠免費。Webhook 驅動的 license 啟動：付完款 1 秒內 sidebar 自動切換顯示帳號 email，不用重啟。
- **資料分析模式** — 內建 `csv_describe` + `python_exec` 與會嵌 Chart.js 的 HTML 報表生成器。
- **人類確認關卡** — `rm`、`sudo`、`mv` 等高風險指令會停下來等你同意。
- **長期記憶** — ChromaDB 記住過去的脈絡，下次規劃時自動帶進來。
- **22 種語言介面** — UI 字串集中在 `static/i18n.js`，build hash + 每 commit cache buster 自動觸發前端重整。
- **設定熱載入** — 模型、threshold、效能開關（streaming / verification / escalation）、API key（Anthropic / Brave Search / Lemon Squeezy）都能在 UI 改完即時生效。

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

## 定價（Open-Core 模式）

OpenTeddy 採用**開放核心**：OSS backend 永遠 MIT 免費，付費的是**polished
desktop 體驗 + 雲端便利功能**：

| | **Free**（這個 repo） | **Lifetime — $99 一次** |
|---|---|---|
| 完整 backend / 工具 / 強化 loop | ✅ | ✅ |
| 自動成長技能 | ✅ | ✅ |
| 22 語系 web 儀表板 | ✅ | ✅ |
| 自己 build desktop | ✅ | ✅ |
| 簽名 .dmg + 自動更新 | ❌ | ✅ |
| 多裝置雲端同步（記憶 / 技能 / 設定）| ❌ | ✅ |
| Premium 技能包（規劃中：Analytics / Marketing / Memory Pro）| ❌ | ✅ |
| 優先 bug 修復 + 私密支援 | ❌ | ✅ |

身分驗證 + 計費完整實作分三階段，code 都在這個 repo：
- **Phase A**：app 啟動匿名 Firebase Auth、寫 `users/{uid}` 紀錄裝置 id
- **Phase B**：Google 登入透過系統瀏覽器 pairing flow（Tauri WebKit 不能直接 popup），用 Cloud Function 簽 customToken 回來、merge 匿名身份
- **Phase C**：Lemon Squeezy webhook → CF 驗 HMAC → 寫 `licenses/{uid}` 跟 `users/{uid}.subscription`，desktop 的 Firestore listener 1 秒內 reflect

OSS user 在 plain browser 跑 — `cloud-sync pill` / `upgrade pill` / `sign-in dialog` 都會自動隱藏，Firebase JS bundle 完全不會載。

## 跑大模型（DGX Spark / 35B 等級）的人請注意

預設啟用的「逐步交付物驗證」是一個額外的 LLM-as-judge call，每個成功 subtask 都會跑一次。在小模型上幾秒就結束沒感覺；但在 DGX Spark + qwen3.5:35b 之類的設定下，每次 5–60 秒，多步驟的報表任務會明顯變慢甚至卡住。

兩種關法：

```bash
# 啟動時 env var
VERIFICATION_ENABLED=false uvicorn main:app --reload

# 或在跑起來之後從 UI 設定面板關掉
# Settings → Per-step deliverable verification → OFF
```

`STREAMING_ENABLED`、`ESCALATION_ENABLED`、`QWEN_NUM_CTX`、`CONTEXT_COMPRESS_AT` 也都是相同模式的熱開關。

## 更完整的文件

更多架構圖、API 列表、環境變數說明、自我成長機制、Claude 介入條件表與 Loop Hardening 機制細節，請看 **[英文版 README](README.md)**，本檔只是精簡入門。

## 支持這個專案

OpenTeddy 是一個人在做的 side project，想證明一套小而開源的組合也能逼近商業代理人的體驗。如果你想支持它繼續長大：

- ⭐ **去 repo 點星** — [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy) — 對我是最大鼓勵。
- 🐛 **開 issue** — 遇到 bug、模型設定卡關都歡迎回報。
- 🧠 **分享你做的技能** — 歡迎 PR。

## 授權

MIT
