<div align="center">

<sub><a href="README.md">English</a> | 繁體中文</sub>

<img src="static/OpenTeddy-logo.svg" alt="OpenTeddy" width="240" />

# OpenTeddy

**在你機器上把事情做完的 AI agent —— 而且重複做的事會越做越熟。**

用 Ollama 在本機執行。只有真的卡住時才升級到雲端模型。重複的工作會變成可重用的技能。

<sub>本機 AI agent · 自主完成工作 · 自我成長的技能 · 混合式升級</sub>

🌐 **官網：** [openteddy.net](https://openteddy.net/) &nbsp;·&nbsp; 📦 **原始碼：** [github.com/m31527/OpenTeddy](https://github.com/m31527/OpenTeddy)

</div>

---

## 你的第一個完成的任務

三個指令。終點是一份做好的產出，不是「裝好了」。

```bash
curl -fsSL https://openteddy.net/install | bash        # Python 3.10+ · 建議裝 Ollama（會拉兩個小模型）
cd ~/OpenTeddy && ./run.sh --open                       # 儀表板在 http://localhost:8000
./openteddy run "分析 ~/OpenTeddy/examples/sales_sample.csv：每月營收趨勢、前三名產品、一張圖表，存成 HTML"
```

終端機會顯示計畫、每一次工具呼叫和結果；報告會落在該 session 的
`reports/` 資料夾，儀表板上會出現 📎 下載晶片。`examples/sales_sample.csv`
隨 repo 附上，所以在接你自己的資料之前就能先跑通。想用瀏覽器？把同一句話
貼進儀表板就行。

這一步**不需要任何雲端金鑰**。之後加一把，OpenTeddy 才會開始在卡住時升級、
並把重複工作長成技能 —— 見「選你的路」。

> 終端機指令速查：[docs/cli.md](docs/cli.md)

## 三個值得在意的理由

| | |
|---|---|
| **本地掌控** | 規劃與執行都在你的硬體上跑。`local_only` 的 session 絕不呼叫雲端 API —— 你的資料、你的模型、你的機器。 |
| **更低的持續成本** | 預設路徑每個 token 都是 $0。只有某一步在本地真的失敗才會叫雲端模型，Usage 頁會告訴你同樣的工作在 GPT-4 上要花多少。 |
| **可重用的技能** | 重複做的事會被寫成純 Python 技能，下一次完全不用推理。技能程式碼由你設定的雲端供應商生成，註冊前會先測試。 |

## 選你的路

| 路徑 | 需要什麼 | 哪些東西會離開你的機器 | 得到什麼 |
|---|---|---|---|
| **純本地試用** | Python 3.10+、[Ollama](https://ollama.ai)、預設 2B 模型約需 8 GB RAM（16 GB 較舒適）。`OPENTEDDY_LLM_MODE=local` 或 Settings → *Local only*。 | **什麼都不會。** | 完整的 agent loop、工具、記憶、儀表板、CLI、排程、Telegram。沒有雲端升級、沒有技能生成。 |
| **混合模式**（預設） | 上述 + **一把**雲端金鑰（Anthropic / OpenAI / Gemini / Deepseek / OpenRouter），貼在 Settings → Cloud LLM Provider。 | 只有本地硬失敗後升級的那些子任務，以及技能生成的提示。`local_only` session 仍然什麼都不送。 | + 自動升級安全網，+ 自我成長的技能。 |
| **開發者 / 手動安裝** | git、venv、`.env` | 同上 | 見下方「快速開始」、Task API、CLI（英文 README）。 |
| **桌面 App（macOS）** | 官網的 `.dmg` | 同上 | 引導精靈幫你裝 Ollama、依硬體等級拉模型。 |

## 三個證明流程

每一個都是真的任務、真的產出。建議照順序試。

**1. 從資料出報告** —— 分析模式

```bash
./openteddy run "分析 ~/OpenTeddy/examples/sales_sample.csv：每月營收趨勢、前三名產品、一張圖表，存成 HTML"
```
`csv_describe` → `python_exec` → `render_chart_report` → `reports/<名稱>.html`，含 Chart.js 圖表與 markdown 摘要。deliverable judge 會確認檔案真的是報告，不是「描述報告的文字」。

**2. 一個寫程式的任務** —— code 模式

```bash
./openteddy run "寫一個把 CSV 轉成 JSON 的 Python CLI（支援 --pretty），加上 pytest 測試，並執行測試" --mode code
```
檔案寫進 session 工作區、測試在 shell 裡跑、摘要會引用測試輸出。高風險指令會在你的終端機停下來問 `y/N`。

**3. 重複的工作變成技能** —— 混合模式

把出報告的任務對不同檔案跑三次。OpenTeddy 會發現重複的模式（ChromaDB 相似度，預設 3 次、0.75），用你的雲端供應商合成技能、先測試、再註冊：

```bash
./openteddy skill list                                   # 新技能以 TESTING 出現，用過幾次升為 ACTIVE
./openteddy skill run <名稱> --input '{"path": "~/data/q3.csv"}'   # 不經規劃、不推理，直接跑技能
```

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
- **`doc_to_markdown` — 8+ 種文件格式直接讀** *(v1.1.0)* — PowerPoint / Word / Excel / EPUB / 圖片（EXIF + OCR）/ 音檔（EXIF + 轉錄）/ HTML / CSV / JSON / XML / ZIP / YouTube URL，全部用一個工具吃，後端是 [markitdown](https://github.com/microsoft/markitdown)。PDF 維持用 pypdf（A/B 測試在履歷/表單 PDF 上勝過 markitdown）。
- **`cyber_skill_lookup` — 755 個專家工作流** *(v1.1.0)* — 索引 [Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)（754 個資安 workflow，映射 MITRE ATT&CK / NIST CSF / D3FEND / ATLAS / NIST AI RMF）+ [last30days-skill](https://github.com/mvanhorn/last30days-skill)（多平台趨勢研究）。Agent 遇到資安 / IR / forensics / 跨平台研究類任務會**先查 catalogue**，拿到 Nitter / Reddit JSON API 等繞過反爬的 workflow 才開始爬，省下「browser_fetch → 被擋 → 失敗」的 10 分鐘無效迴圈。
- **斷線可恢復的 WS** — 600 筆 ring buffer + `?since=` 重播，網路抖一下、刷新分頁都不會卡 UI。
- **技能會自己長出來** — 透過 ChromaDB embedding 偵測重複的 goal pattern（不依賴小模型自我反思），當過去有 N 個語意相近的 goal 達標時自動合成技能名稱+描述，交給 Claude 寫成 Python 函式。可在 Settings 調整門檻（預設 3 次重複、相似度 0.75）。
- **Web 儀表板** — 提交任務、即時看工具呼叫、審核敏感指令、管理記憶、即時顯示「已省 $X vs GPT-4」，還有 GFM 表格 + Chart.js 數值標籤 HTML 報表。
- **Capabilities tab** — 內建 Tools 跟自動長出來的 Skills 合併在同一個可篩選列表，type badge 標明來源；技能用越多自動從 TESTING 升 ACTIVE。
- **上傳就能列印** — 在對話或 Telegram 上傳檔案，說「列印」或「印 2 份 雙面」，大約一秒就送到印表機（透過 CUPS），回覆附工作編號和碳粉不足之類的提醒。Word / Excel / PowerPoint 會先用 LibreOffice 轉成 PDF；「整理成題庫再印出來」這種要先做事的，由 agent 做完再呼叫列印工具。詳見 [列印](#列印)。
- **和 Claude Code 雙向合作** — `openteddy mcp` 讓 Claude Code 可以使用 OpenTeddy（控制家裡設備、列印、傳 Telegram 給你、叫用 agent）；在手機上說「用 Claude Code 在 ~/projects/shop 修好失敗的測試」，就會在背景交給 Claude Code，完成後把結果和改動的檔案傳回來。詳見 [Claude Code ↔ OpenTeddy](#claude-code--openteddy)。
- **每個人各自的個人助理** — 「今天有什麼行程」、「有什麼新信」、「明天早上 9 點提醒我繳費」和每天早上的「早安簡報」（天氣、往台灣來的颱風、行程、提醒、未讀信、離線設備）都不用經過模型。每個人在自己的 Telegram 私訊連結自己的 Google 帳號（OAuth、唯讀），只有本人拿得到自己的資料；含有這些資料的回答不會被保存，也不會顯示在網頁上。提醒也可以由小愛音箱唸出來。詳見 [個人助理](#個人助理行事曆gmail提醒早安簡報)。
- **透過 Home Assistant 控制家裡的設備** — 「關客廳燈」、「冷氣開 26 度」、「家裡有哪些設備離線」大約一秒由 Home Assistant 完成；它能接米家、Google TV 和兩千多種品牌。小愛音箱可以播報、定時提醒、放音樂，也能代你下語音指令控制紅外線家電。門鎖、保全、車庫門會等你回覆「確認 代碼」才動作。詳見 [智慧家庭](#智慧家庭home-assistant)。
- **從 Telegram 驅動 OpenTeddy** — 在 Settings → Notification Credentials 填一支 bot token + 白名單 chat_id 後，手機上就能用 Telegram 下任務：發任何文字當 goal，也可以傳文件或照片（存進 session 的 `uploads/`，給下一句話用）；處理時會出現一個小小的「Thinking」動畫，回答會排版成正常的文字而不是一堆 `###`、`**` 符號，另外附上 `📎 Files produced`（文字檔直接 inline 顯示，二進位走 sendDocument 變附件）。每則訊息都看得到最近幾輪對話。自動 approve 高風險工具讓日常任務不卡 — 但 hard denylist（`rm`、`rmdir`、`DROP TABLE`、`TRUNCATE TABLE`、`DELETE FROM`、`mkfs`、`dd if=…/of=/dev/…` 等）絕對擋住破壞性指令。命令：`/start`、`/help`、`/cancel`、`/new`。詳見 [遠端存取](#遠端存取手機--telegram--tailscale)。
- **手機友善的 Web UI 透過 Tailscale** — `./run.sh --host 0.0.0.0` + 自家 tailnet 就讓 dashboard 在手機瀏覽器上能用。Sessions / chat / artifact 預覽都 responsive；手機寬度時 session header 會自動把記憶 / 隱私 / 匯出收進右上 ⋯ kebab。不用 port forward、不用 nginx、不用公網 DNS — 手機裝 Tailscale app 直接連 `http://<你的機器名>:8000`。
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

- Python 3.10+
- [Ollama](https://ollama.ai)：
  ```bash
  ollama pull gemma4:e2b
  ollama pull qwen3.5:2b
  ```
- （可選）一把 Anthropic / OpenAI / Gemini / Deepseek / OpenRouter 金鑰 —— 只用於升級與技能生成，之後在 Settings → Cloud LLM Provider 填也可以

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
# 純本地：不用再填任何東西。OPENTEDDY_LLM_MODE=local 可保證完全不呼叫雲端。
# 混合模式（預設）：加「一把」雲端金鑰，例如 ANTHROPIC_API_KEY=…，或之後在
#                   Settings → Cloud LLM Provider 貼上。只用於升級與技能生成。
```

### 4. 啟動

最方便的方式是用內建的 `./run.sh`——會自動 activate `.venv`、檢查 Ollama、跑 `uvicorn main:app --reload`：

```bash
./run.sh                    # 預設只綁 127.0.0.1:8000（安全預設）
./run.sh --open             # 同時開瀏覽器
./run.sh --port 8001        # 換 port
./run.sh --host 0.0.0.0     # ⚠ 開放給整個區網 / Tailscale / 其他機器
./run.sh --no-reload        # production 風格——不監看檔案改動
./run.sh --help             # 看完整 flags
```

> ⚠️ **`--host 0.0.0.0` 等於把 agent 開放給所有能連到該 port 的機器**。Agent 有 `shell_exec_write` / `delete_file` 之類強力工具，只有在你信任網段上每一台機器（私人家用網段 / Tailscale tailnet / 防火牆後的專屬伺服器）才用 `0.0.0.0`。公網伺服器請放在 nginx / Caddy / Cloudflare Tunnel 後面並加上認證。**想用手機操作 OpenTeddy 的話，推薦做法是 `--host 0.0.0.0` + Tailscale — 詳見 [遠端存取](#遠端存取手機--telegram--tailscale)。**

不想用 `run.sh`？直接：

```bash
uvicorn main:app --reload
# 儀表板： http://localhost:8000
# API 文件： http://localhost:8000/docs
```

## Linux 伺服器第一次設定（DGX / Jetson / 無螢幕節點） *(v1.1.5)*

如果你要把 OpenTeddy 部署在 Linux 伺服器（NVIDIA DGX Spark、Jetson、Raspberry Pi 5、機房裡的機架機，任何沒有固定螢幕的環境），除了 `uvicorn main:app` 之外還需要：

- 一支 **Chromium 系瀏覽器**用 systemd 跑著，這樣 OpenTeddy 的 [`chrome_attached_tool`](tools/chrome_attached_tool.py) 才能透過 CDP 抓需要登入的網站（X / Twitter、LinkedIn、付費 SaaS、公司內網 wiki）
- 一次性的**手動登入**，讓 headless 瀏覽器帶著你的真實 session — `x_search('黴菌', top_n=10)` 在沒登入的 browser 上回的是空的
- 一套 **profile 目錄 + systemd unit**，這樣重開機也還在

[`scripts/`](scripts/) 底下三支 script 把整個流程包起來：

| Script | 什麼時候跑 |
|---|---|
| `scripts/quickstart.sh --login` | **新機器只跑一次**。一行指令搞定下面所有事 |
| `scripts/setup-edge-cdp.sh` | quickstart 內部會呼叫。裝 Brave（ARM64）或 Microsoft Edge（amd64）+ 寫 systemd unit |
| `scripts/login-helper.sh` | 之後 cookies 過期（X 約 30 天、LinkedIn 約 90 天）就跑一次。會跳 GUI 視窗讓你登入 |

### 一行指令搞定新機器

在有 GUI 桌面的環境（AnyDesk、實體螢幕、`ssh -X`、GNOME/KDE）：

```bash
git clone https://github.com/m31527/OpenTeddy
cd OpenTeddy
bash scripts/quickstart.sh --login
```

它會：

1. **裝好 CDP 瀏覽器** — 自動偵測 host arch：
   - `aarch64` / `arm64` → **Brave Browser**（官方 ARM64 apt repo、無沙箱限制）
   - `x86_64` → **Microsoft Edge**（官方 amd64 apt repo）
2. **寫好 systemd unit**（`openteddy-cdp.service`）讓 browser headless 跑在 `127.0.0.1:9222`，`Restart=always`，重開機自動拉起來
3. **跳出 GUI 瀏覽器視窗**讓你登入 X / LinkedIn / 任何需要的網站。關掉視窗 → script 自動切回 headless 模式，cookies 保留在 profile 裡
4. **建 Python venv** + `pip install -r requirements.txt`
5. **啟動 OpenTeddy backend** 跑在背景 `:8000` 並做 healthcheck

完工之後 `http://<host>:8000` 就是完整可用的 OpenTeddy。在 chat 打「整理 X 上最近討論黴菌的熱門推文 top 10」， [`x_search`](tools/chrome_attached_tool.py) 工具會用你登入的 session 拉真實推文。

### 如果只能純 SSH（沒 GUI 桌面）

兩條乾淨的路：

- **`ssh -X admin@<host>`** — Brave 視窗會透過 X11 forwarding 投影到你筆電的螢幕上。前提是筆電有 X11（Linux 原生有、macOS 要裝 [XQuartz](https://www.xquartz.org/)）
- **`scripts/setup-novnc-login.sh`** — 在 server 上裝 Xvfb + noVNC，用網頁瀏覽器登入：
  ```bash
  sudo bash scripts/setup-novnc-login.sh
  sudo systemctl start openteddy-novnc-login.service
  # 在你筆電上：
  ssh -L 6080:localhost:6080 admin@<host>
  # 開瀏覽器 http://localhost:6080/vnc.html
  # 在跳出來的 Brave 視窗登入 → 關 tab → 在 server 跑：
  sudo systemctl stop openteddy-novnc-login.service
  ```
  noVNC 預設只 bind 在 `127.0.0.1`，靠 SSH tunnel 做認證 + 加密。要對 LAN 開放（Tailscale mesh 等場景）就設 `OPENTEDDY_NOVNC_BIND=0.0.0.0`，setup script 安裝結束時會提醒安全 trade-off

### cookies 過期了怎麼補登

`x_search` 開始回空的 `posts: []`、或者排程任務開始拿空結果，就代表 cookies 過期了。跑：

```bash
bash scripts/login-helper.sh
```

它會暫停 headless service、開 GUI Brave 並導到 `https://x.com/login`、等你登入完關掉視窗、自動把 headless service 拉回來。大概 2 分鐘。

### Fleet 部署（5-10 台節點）

同樣三支 script 直接當 Ansible playbook payload — 每台節點都跑一樣的安裝、每台都被 trusted operator 個別登入一次。我們在 NVIDIA DGX Spark fleet 用的模式：

```bash
# 每台節點一次性（cloud-init / Ansible 跑）
ansible all -m shell -a "git clone https://github.com/m31527/OpenTeddy /opt/openteddy"
ansible all -m shell -a "bash /opt/openteddy/scripts/quickstart.sh"

# 然後 operator 從 AnyDesk / VNC 進每台跑：
#   bash /opt/openteddy/scripts/login-helper.sh
```

每台節點擁有自己的 browser profile + cookies；不必跨節點同步。如果要做「一次登入、全 fleet 共用」，把 Playwright 格式的 `storage_state.json` 放到 `/var/lib/openteddy/storage_state.json`，`chrome_attached_tool` 會在每次 attach 時自動注入。詳見 [`scripts/capture-edge-state.md`](scripts/capture-edge-state.md)。

### 更新與部署到另一台機器

服務不會自動重新載入，每次更新後要重啟。`openteddy update` 一次做完：`git pull` → 安裝缺少或有變動的套件 → 重啟服務 → 健康檢查。現在每次更新都會補裝缺少的套件，不只在 `requirements.txt` 有變動時才裝，所以曾經漏裝套件的環境不會一直壞下去。

從開發機部署到另一台（先 `git push`，目標機會從 GitHub 拉）：

```bash
./deploy.sh user@host[:/path]     # 在目標機 pull → 套件 → 重啟 → 健康檢查
./deploy.sh user@host --autostash # 目標機有本地修改，而這次更新剛好動到同一個檔案
```

目標機上的本地修改會保留。詳見 [docs/cli.md](docs/cli.md)。

### 排程沒有執行？

排程需要 `APScheduler` 套件。`GET /health` 會回報 `"scheduler": "running" | "stopped" | "disabled"`，`openteddy health` 在停用時會提醒你。排程功能停用時，你說「每天 7:40 提醒我…」會直接得到修正方法，不會再交給 agent 自己發揮：曾經有 agent 寫了一支沒有任何東西會執行的腳本，然後回報「已排好」。

### 機器很熱或自己斷電

DGX Spark / GX10 這類小主機在長時間高負載下可能自己斷電，系統紀錄會直接中斷，沒有關機流程也沒有錯誤訊息，因為是韌體切的電。要知道斷電前發生什麼事，可以開飛行紀錄器，它每 2 秒把 GPU 溫度與功耗、最熱的溫度區、可用記憶體、負載和最忙的程式寫進硬碟：

```bash
systemd-run --user --unit=power-recorder --collect ./scripts/power_recorder.sh   # 執行到重開機為止
tail -30 ~/power-recorder.csv                                                     # 斷電後查看
```

在一台機器上實際查到的原因：

- **一直重啟的服務**：遺留的 `openteddy-vllm` 服務重啟了 126,000 次，每 14 秒初始化一次 GPU。用 `systemctl list-units --failed` 和 `NRestarts` 檢查。
- **背景呼叫模型的程式**：定時呼叫 Ollama 的程式（交易機器人、agent 的心跳）會讓 GPU 全天滿載。Ollama 的紀錄會列出每次請求的來源位址和耗時。
- **每個模型只用一種 `num_ctx`**：用不同 context 長度呼叫同一個模型，每次切換都會重新載入。

## 遠端存取（手機 / Telegram / Tailscale）

兩條互補的路徑可以在不在主機旁邊時操作 OpenTeddy。兩者打到同一個 server、同一份 sessions —— 你可以在通勤時用 Telegram 開一個 goal，回家後在桌面 web UI 看完整 artifact。

### A. 用 Tailscale 從手機開瀏覽器連 Web UI

最簡單的「在哪都能瞄一下 agent」設定。不用 port forward、不用 DNS、不用 nginx。

1. **server 端**（OpenTeddy 跑的那台）：
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ./run.sh --host 0.0.0.0
   ```
   `--host 0.0.0.0` 讓 uvicorn 綁所有 interface（包含 tailnet）。實際擋外部存取的是 **Tailscale 本身** — 只有你 tailnet 上的裝置連得到。

2. **手機**：App Store / Play Store 裝 Tailscale，登入同一個帳號，打開 VPN toggle。

3. **開瀏覽器**連 `http://<機器名>:8000`（或 `tailscale status` 看到的 IP）。Web UI 用同一份 sessions、同一份 artifact chip、同一份 WebSocket live stream。手機寬度下，session header 自動把記憶 / 隱私 / 匯出收進右上 ⋯ kebab。

> ⚠️ **為什麼推薦 Tailscale 而不是裸的 `--host 0.0.0.0` 在 LAN？** 你家裡 WiFi 上的訪客裝置也算 LAN，而 agent 有 `shell_exec_write` 等強力工具。Tailscale ACL 才能精準限定「只有我授權過的裝置進得來」。如果你真的要走 LAN，請只在你 100% 信任的家用網段用。

### B. 從 Telegram 雙向操作 OpenTeddy

從任何地方傳 goal，結果推回同一個 chat。即時進度會在**同一則訊息上 edit**，不會洗版。設計給 24/7 跑著的 self-hosted server（Mac mini、NUC、家用 Linux box）— 桌面 app 關掉就會停止 polling，所以**桌面 app 不適合**這個功能。

#### 1. 建一支 bot

打開 Telegram → 找 **@BotFather** → `/newbot` → 跟著步驟走。記下 bot token（長得像 `123456:ABC-DEF1234...`）。

#### 2. 找你的 chat_id

傳任何訊息給 **@userinfobot**，它會回你一個數字 `id`（例如 `987654321`）。如果是群組：在群組裡發訊，然後 forward 那則到 **@userinfobot**，它會給群組 chat_id（負數，例如 `-1001234567890`）。

#### 3. 先跟自己的 bot 開啟對話

在 Telegram 搜你 bot 的 `@username`，按 **Start**（或送 `/start`）。**這步最常被忽略**。沒做的話 Telegram 的「bot 不能無預警 DM 你」規則會擋下所有 outbound 測試，你會看到 `chat not found` 錯誤。

#### 4. 設定 OpenTeddy

在 **Settings → Notification Credentials**：

| 欄位 | 填什麼 |
|---|---|
| Bot Token | BotFather 給的 token |
| Default Chat ID | 你的數字 chat_id（讓 `telegram_send` 工具能用） |
| **Test ping** 按鈕 | 按下去，預期回 ✓ + Telegram 收到「🐻 OpenTeddy ping」 |
| **Enable inbound polling** | ✅ 勾起來 |
| Chat-ID whitelist | 你的 chat_id（多個用逗號分隔） |

存檔。**重啟 server**（hot-reload 還沒做 — 目前 `./run.sh` 要手動重啟才會啟動 polling loop）。重啟後 log 應該看到：

```
[INFO] telegram_bridge: Telegram inbound bridge started — polling with 1-id whitelist.
```

如果看到 `Telegram inbound bridge NOT started: …` 的話，訊息會告訴你是哪個欄位沒設好。

#### 5. 從 Telegram 操作 agent

| 你發 | 會怎樣 |
|---|---|
| 任何文字 | 當 goal 跑，在這個 chat 綁的 session 裡，跑完推結果回來 |
| 文件或照片 | 存進 session 的 `uploads/`，附在下一則訊息（或照片說明）上，例如接著說「幫我總結」或「列印」 |
| 傳檔後說「列印」、「印 2 份 雙面」 | 直接列印，見 [列印](#列印) |
| 「關客廳燈」、「家裡有哪些設備離線」 | 由 Home Assistant 回答，見 [智慧家庭](#智慧家庭home-assistant) |
| 「確認 1234」 | 執行先前被暫停、等你確認的門鎖 / 保全 / 車庫門動作 |
| `/start` | 確認連線 |
| `/help` | 命令列表 |
| `/cancel` | 中止當下正在跑的 task |
| `/new` | 換新 session（舊的保留在歷史） |

處理超過 2 秒時會出現一個循環播放的「Thinking」小貼圖，答案送出前自動刪除。Agent 回覆包含：

- **答案在最前面** — Markdown 會轉成 Telegram 的格式（粗體、清單、連結、程式碼；標題變粗體、表格變條列），太長會自動分成多則，最後一行是淡淡的「✅ 完成 · 11.7 秒」
- **`📎 Files produced`** — 每個 artifact 的檔名 + 大小 + 是哪個工具產的（包含 shell redirect 寫出來的檔，已被後台 workspace scanner 抓到）
- **內容 inline** — 小於 3 KB 的文字檔直接當下一則訊息丟過來；大檔 / 二進位走 Telegram `sendDocument` 變成「附件 card」，你 tap 就下載

#### 安全模型

- **硬性白名單**：不在白名單的 chat_id 來訊**直接沈默丟掉**（不留 probe signal）。白名單空 = inbound 連啟動都不啟動，就算 toggle 開了也一樣。
- **Auto-approve 高風險工具**：你白名單的 chat_id 本身就是 consent signal — `shell_exec_write` / `python_exec` / `file_write` 都不會卡 approval prompt（反正你也不在 web UI 旁邊）。
- **Denylist 硬擋破壞行為**：無論 approve 與否，tool registry 會擋 `rm` / `rmdir` / `unlink` / `git rm` / `shred`、SQL `DROP TABLE` / `DROP DATABASE` / `TRUNCATE TABLE` / `DELETE FROM`、系統層 `mkfs` / `dd if=…/of=/dev/…` / `> /dev/sd[a-z]` / `fdisk` / `format X:` / 遞迴 `chmod 0…`，以及任何工具名含 `delete` / `remove` / `drop_table` / `truncate` / `wipe` / `purge`。被擋會明確告訴你「open the web UI 手動 approve」。
- **日誌不會留下 token**：HTTP 套件的每筆請求不再寫進日誌，每一行日誌和錯誤追蹤裡的 Telegram bot token、`?key=` / `?token=` 參數和 Bearer token 都會遮蔽，系統日誌和診斷用的日誌緩衝都一樣。
- **10 分鐘 hard timeout**：每個 Telegram-driven run 用 `asyncio.wait_for(timeout=600)` 包住，Ollama 掛掉、tool deadlock 都不會把 chat 卡死 — 10 分鐘整 bot 會回「⌛ Task ran longer than 10 min and was force-cancelled」而不是無聲沈默。

#### 排查

```bash
curl -s http://<server>:8000/admin/telegram/status | jq
```

回 bridge 當下的狀態 — `running`、`inbound_enabled`、`token_set`、白名單實際內容、**最近一次被沈默丟掉的 chat_id**（這個是「為什麼我發訊息沒回」的最快答案），以及哪些 chat 正在跑 task。Token 不會回傳，只回一個 bool flag 表示有沒有設。

## 列印

在對話視窗或 Telegram 上傳檔案，說要列印就好。單純的列印要求不經過規劃模型：附檔直接交給 CUPS（`lp`），大約一秒就回覆。

| 你說 | 會怎樣 |
|---|---|
| 「我要列印」、「print this」 | 印出這則訊息附的檔案 |
| 「幫我印 2 份 雙面」、「印第 1 到 3 頁 黑白」 | 份數、雙面 / 單面、黑白、頁數範圍 |
| 沒附檔直接說「列印」 | 印這個對話兩小時內最近上傳的檔案 |
| 「把題庫整理好再印出來」 | 不是單純列印：agent 先做事，再呼叫 `print_file` 工具 |
| 「碳粉還有嗎」 | 問印表機狀態，不會列印：agent 可以呼叫 `printer_status` |

```
印表機：Brother HL-L3270CDW series
🖨️ 已送出列印：考卷.pdf（3 頁，2 份，雙面），工作編號 42
⚠️ 黑色碳粉剩 10%，快用完了
```

**支援的格式：** PDF、圖片（自動縮放到一頁）、文字、Markdown、CSV 直接印；Word、Excel、PowerPoint、RTF、HTML 先用 LibreOffice（`soffice`）轉成 PDF。

**用哪台印表機：** Settings → 🖨️ 印表機 → 預設印表機。留空就用 CUPS 的預設，或這台機器唯一的印表機。同一個區塊會顯示印表機狀態和碳粉存量，這是 OpenTeddy 直接透過 IPP（`ipptool`）問印表機的。`GET /printers` 也會回傳一樣的資訊。

**建立永久佇列：** Linux 上的網路印表機通常是自動出現的「暫時佇列」，印表機關機時會消失，這時送出的列印就會失敗。改用永久佇列的話，工作會排隊，印表機開機或醒來後自動印出：

```bash
lpinfo --include-schemes dnssd -v          # 找到印表機的 dnssd:// 位址
lpadmin -p Office_Printer -E -v 'dnssd://…' -m everywhere -o printer-error-policy=retry-job
```

再到 Settings → 印表機選它。帳號在 `lpadmin` 群組的話不需要 sudo。

**保護措施：** 只能列印 session 工作區裡的檔案，agent 沒辦法把機器上任意的檔案印出來。份數有上限（Settings →「每次最多份數」，預設 20），一次最多 500 頁。排程任務不會自己列印。

## 智慧家庭（Home Assistant）

OpenTeddy 不直接接各個品牌，而是交給 [Home Assistant](https://www.home-assistant.io/)：它負責米家、Google TV / Chromecast、Philips Hue、Tuya 等兩千多種整合，OpenTeddy 只用一組存取權杖跟 Home Assistant 溝通。OpenTeddy 停機時，Home Assistant 的 App 和自動化照樣能用。

```
Telegram / 對話 / 排程 ──▶ OpenTeddy ──▶ Home Assistant ──▶ 米家 · 小愛音箱 · 電視 · …
```

### 設定步驟

1. **安裝 Home Assistant**，裝在跑 OpenTeddy 的同一台（Docker；`--network host` 才找得到區網裡的設備）：

   ```bash
   docker run -d --name homeassistant --restart unless-stopped --network host \
     -e TZ=Asia/Taipei -v ~/homeassistant:/config ghcr.io/home-assistant/home-assistant:stable
   ```

   打開 `http://<機器>:8123`，建立管理員帳號。
2. **加入你的品牌**：Home Assistant 的「設定 → 裝置與服務」。米家請用 [Xiaomi Miot Auto](https://github.com/al-one/hass-xiaomi-miot)（把 `custom_components/xiaomi_miot` 放進 `~/homeassistant/custom_components/` 再重啟）。它支援 `cn`、`tw`、`sg`、`us`、`de`、`ru`、`i2` 伺服器，也能同時登入多個帳號：設備分在台灣和中國兩個帳號的話，每個地區各加一次。連線模式請選**雲端**：本機模式會讓大部分大陸設備（包括小愛音箱）顯示無法使用。小米官方的 Xiaomi Home 整合沒有列出台灣。
3. **把設備名稱改成你平常說的**（客廳燈、臥室冷氣），或加別名。短指令是靠名稱找設備的。
4. **建立長期存取權杖**（個人資料 → 安全性），貼到 OpenTeddy 的 Settings → 🏠 Home Assistant，儲存後按「測試連線」。會顯示版本、設備數量，以及哪一台是小愛音箱。`GET /home/status` 也會回傳一樣的資訊。

### 可以怎麼說

| 你說 | 路徑 | 會怎樣 |
|---|---|---|
| 「關客廳燈」、「冷氣開 26 度」、「電視音量調小」 | 直接，約 1 秒 | 交給 Home Assistant 的 Assist 執行；它聽不懂時改由 agent 用 `home_*` 工具處理 |
| 「家裡有哪些設備離線」、「有設備斷線嗎」 | 直接，不到 0.1 秒 | 列出離線設備，一台設備一行，附房間和離線多久 |
| 「家裡有哪些設備」 | 直接 | 按房間列出所有設備和在線狀態 |
| 「叫小愛說晚餐好了」、「請小愛提醒出門」 | 直接，約 1 秒 | 小愛音箱把這句話唸出來（`xiaoai`，模式 `say`） |
| 「叫小愛打開冷氣」、「跟小愛說打開電視」 | 直接，約 1 秒 | 像你親口對小愛說一樣（模式 `command`），可以控制只有小愛能控制的紅外線家電 |
| 「播放音樂」、「放點輕音樂」、「我想聽周杰倫的歌」、「小米音箱下一首」、「小愛音量調到 30」、「停止音樂」 | 直接，約 1 秒 | 當成語音指令交給小愛，由它自己的音樂服務播放 |
| 「請小愛音箱週一到週五早上 7:40 提醒出門」 | 排程 | 建立平日排程，每次執行都直接送到音箱，不經過模型 |
| 「我出門了，把該關的都關掉」 | agent | `home_overview` / `home_devices` 找設備，再逐一 `home_control` |
| 「每天早上 9 點檢查有沒有設備離線，有才通知我」 | 排程 | 每天檢查，有設備離線才傳 Telegram |
| 「確認 1234」 | 直接 | 執行先前被暫停、等你確認的動作 |

音箱可以叫「小愛」、「小愛同學」、「小愛音箱 Pro」、「小米（AI）音箱」、「智慧音箱」或「音箱」。給小愛的指令必須是中文，所以英文的播放、停止會自動轉成中文。說到「每天」、「週一」、「明天」這類重複或未來的時間，不會立刻播放，而是建立排程。提到電視，或「我要製作睡眠音樂」這種創作要求，不會送到音箱。

提到設備或音箱的訊息一定保留工具：Assist 處理不了時，交給有 `home_*` 和 `xiaoai` 工具的 agent，絕不會落到沒有工具的聊天模式，只回你一句「我沒辦法控制設備」。

「離線」的判斷：一台設備的所有項目都沒有實際狀態，而且至少一項無法使用才算。按鈕類的項目永遠是 `unknown`，雲端整合也會讓正常設備的個別功能顯示無法使用，這些都不算離線。

### 安全

- **確認碼：** 門鎖、保全、水閥、警報器、熱水器，以及車庫門、閘門、門這類捲簾，永遠不會直接執行。動作會先暫停，回傳一組四位數代碼；只有**你自己傳一則新訊息**「確認 1234」才會執行，5 分鐘內有效、只能用一次。模型不能自己確認，Telegram 的自動核准也不適用，排程同樣不能確認。名單在 Settings →「需要確認的設備類型」。
- **敏感字詞不走快速指令**（鎖、保全、瓦斯、車庫、lock、alarm、garage…），多步驟的要求也不走。
- **Home Assistant 自己的公開設定仍然有效：** Assist 只會操作「設定 → 語音助理 → 公開」裡的設備，門鎖預設不公開。不想被語音控制的設備，請在那裡確認。
- 權杖以密碼欄位儲存，匯出診斷資料時會遮蔽。

## 個人助理：行事曆、Gmail、提醒、早安簡報

每個人在**自己的 Telegram 私訊**裡連結**自己的** Google 帳號，只有本人拿得到自己的行程和信件。OpenTeddy 也會記一次性的提醒，每天早上把今天要知道的事整理成一則訊息。Google 的存取都是**唯讀**：不會新增、寄出、標成已讀或刪除任何東西。

### 隱私怎麼保護

OpenTeddy 沒有使用者帳號，唯一知道「誰在問」的入口是 Telegram 私訊（由 bridge 在程式內部標記，HTTP 請求偽造不了），所以一個私訊聊天就代表一個人。

- 網頁、CLI、MCP、群組都不代表任何人：行事曆和 Gmail 一律拒絕，直接回答和 agent 的工具都一樣。
- 讀過某人行事曆或信件的任務會標成**私人**：存起來的回答、網頁上的即時事件都換成一行說明，記憶也不保留，只有 Telegram 的回覆有完整內容。
- refresh token 加密存放（資料庫旁的 `google_tokens.key`，權限 600），任何地方都不會顯示。
- 提醒、以及早安簡報裡的提醒，只屬於設定它的那個對話。
- 網頁介面本身還是沒有登入：打得開網頁的人看得到其他對話中非個人的紀錄，請只放在區網或 tailnet 裡。

### 設定步驟：管理者做一次

1. [Google Cloud 主控台](https://console.cloud.google.com/) → 建立專案 →「API 和服務」→ 啟用 **Google Calendar API** 和 **Gmail API**。
2. 「OAuth 同意畫面」→ 外部 → 加入 `calendar.readonly` 和 `gmail.readonly` 兩個範圍 → 發布狀態設成**正式版**。「測試」狀態下 Google 會讓每個 refresh token 7 天就過期。應用程式維持未驗證即可，家庭使用沒問題（上限 100 個帳號）。
3. 「憑證」→「建立憑證」→「OAuth 用戶端 ID」→ 類型選**電腦版應用程式**。把用戶端 ID 和密鑰貼到 **Settings → 📅 Google**，儲存後按「測試連線」（`GET /personal/status`：是否設定好、有幾個人連結，只有人數，不顯示是誰）。

### 連結帳號：每個人自己做

1. 在你和 bot 的 Telegram 私訊說「**連結我的 Google**」。
2. 打開連結、登入、兩項都勾選允許。Google 會提醒這個應用程式未經驗證：按「進階」→「前往…」。如果 Telegram 內建瀏覽器顯示已封鎖，改用 Safari 或 Chrome 開啟。
3. 最後瀏覽器會停在一個打不開的頁面（`127.0.0.1:8749`）。**把那一頁的網址複製起來，貼回聊天裡**就完成了。Google 不接受區網位址當回呼網址，所以用這個方式在哪裡都能完成，不用改網路設定。連結只給你本人、15 分鐘內有效、只能用一次。

說「Google 連結了嗎」可以看連結的帳號；說「解除 Google」會在 Google 撤銷授權並刪除。也可以到 Google 帳戶 → 安全性 → 第三方存取權移除。

### 可以怎麼說（在你的 Telegram 私訊）

| 你說 | 路徑 | 會怎樣 |
|---|---|---|
| 「今天有什麼行程」、「明天的會議」、「這週行程」、「下週有哪些會議」 | 直接 | 列出你在 Google 日曆顯示的所有日曆的行程、時間和地點，已拒絕的不列 |
| 「有什麼新信」、「有重要的信嗎」 | 直接 | 主要分類 24 小時內的未讀信：寄件人、主旨、時間 |
| 「明天早上 9 點提醒我繳電話費」、「30 分鐘後提醒我關火」、「下週一提醒我交報告」 | 直接 | 建立一次性提醒，時間到傳到你發問的那個聊天 |
| 「下午 3 點叫小愛提醒我出門」 | 直接 | 下午 3 點小愛音箱說出來，Telegram 也會收到 |
| 「我有哪些提醒」、「取消提醒 3f2a1b」、「取消剛剛的提醒」 | 直接 | 只看得到、取消得了你自己的提醒 |
| 「早安簡報」 | 直接 | 天氣（要不要帶傘）、往台灣來的颱風（沒有就不提）、你今天的行程和提醒、你的未讀信、家裡離線的設備 |
| 「每天早上 7:30 給我早安簡報」 | 排程 | 每天早上把簡報傳到 Telegram（在你的私訊裡設定，就會包含你的行程和信件） |
| 「看我明天的行程，空檔排一個健身」、「整理今天未讀信的重點」、「會議前 15 分鐘提醒我」 | agent | 用 `calendar_events`、`gmail_unread`（Gmail 搜尋語法）、`reminder_create` 當成大任務的其中幾步 |

提醒存在資料庫裡，重開機不會不見；OpenTeddy 停機時到期的提醒，開機後照樣送出，並註明晚了多久（超過 30 分鐘就不叫小愛唸了）。「每天」、「每週一」這種重複的說法會變成排程；「如果明天下雨…」這種有條件的交給 agent；已經過去的時間會先問你，不會直接設定。簡報的每個部分都是選配，而且同時抓取；其中一項失敗只會顯示一行說明，其他照常送出。

## Claude Code ↔ OpenTeddy

Claude Code 是比較強的工程師，OpenTeddy 是一直在線、連著實體世界的管家。兩者可以雙向合作。

### 在 Claude Code 裡使用 OpenTeddy（MCP）

`openteddy mcp` 會把 OpenTeddy 變成一個 MCP 伺服器。它只是連到執行中 OpenTeddy 的輕量程式，所以任何連得到的電腦都能用，例如筆電連家裡的伺服器：

```bash
claude mcp add openteddy -- /path/to/OpenTeddy/openteddy --url http://<執行 OpenTeddy 的機器>:8000 mcp
```

| 工具 | Claude Code 可以拿來做什麼 |
|---|---|
| `ask_openteddy` | 任何要求：問 agent、做報表、說「每天早上 8 點…」建立排程 |
| `home_control` | 「關客廳燈」、「冷氣開 26 度」、「叫小愛說：部署完成了」、「家裡有哪些設備離線」 |
| `print_file` | 列印本機檔案（先上傳），可指定份數、雙面、黑白、頁數 |
| `notify_me` | 傳 Telegram 訊息給你，例如「部署完成，測試全過」 |
| `weather` | 天氣與颱風 |
| `self_review` | OpenTeddy 的自我體檢 |
| `list_schedules` | 排程和下次執行時間 |

這些要求都在同一個叫「Claude Code」的 OpenTeddy 對話裡執行，前後文會延續。Telegram 通知需要在 Settings 設好 bot token 和 Default Chat ID。

### 從 OpenTeddy 把寫程式交給 Claude Code

在 Telegram、網頁或排程裡說：

```
用 Claude Code 在 ~/projects/shop 修好失敗的測試
Claude Code 幫我 review 最近的改動          ← 用預設資料夾
Claude Code 做到哪了                        ← 查進度
```

OpenTeddy 會在自己那台機器的那個資料夾，於背景執行 `claude -p`，並立刻回覆。完成後你會收到 Telegram 訊息，內容有 Claude Code 自己寫的總結、改了哪些檔案、花了多久；結果也會存進對話，之後問「剛剛改了什麼」也答得出來。規劃模型也可以用 `claude_code` 工具（高風險）把工作交出去，用 `claude_code_jobs` 查狀態；`GET /claude-code/jobs` 會列出所有工作。

界線：

- **可以直接改檔案，指令只限白名單**：git status / diff / log / show、ls / cat，以及常見的測試指令（`pytest`、`npm test`、`go test`、`cargo test`、`make test`）。
- **絕不** `git push`、`git reset --hard`、`rm -rf`、`sudo`，也**不會 commit**，改動會留著讓你檢查。
- 只能在那台機器上已存在的資料夾工作，不能動 OpenTeddy 自己的程式碼；超過 30 分鐘會自動停止（可在 Settings 調整）。
- 執行 OpenTeddy 的那台機器要安裝並登入 Claude Code。它用的是你的 Claude 方案或 API 金鑰，不是地端模型。

## Fleet — 分散式「AI 大腦群」（可選）

把多台 OpenTeddy 串成一個叢集：**中控**節點派任務給 **worker** 節點（每台用自己的模型 + 工具 + 資料執行），worker 的 watcher 迴圈發現異常時還會**主動推 alert**。為 5–10 台 NVIDIA DGX Spark 設計，但任何能互連的安裝都能用。

**預設關閉 — 對單機版零影響。** 沒設 `OPENTEDDY_FLEET_ROLE` 就不會 import `fleet/` 任何東西。桌面 / 個人用戶完全不用碰，沒有任何 fleet 的 `.env` 是必填的。

### 先在單機試（不用第二台機器）

```bash
bash scripts/fleet-demo.sh   # 同一台機器內起中控+worker → 🎉 PASSED
```

### 真多機 — 每個角色一份 `.env`

**步驟 1 — 產一個共用 token，每台填一樣的值：**

```bash
openssl rand -hex 32          # 複製輸出
```

**步驟 2 — 中控節點（只有一台）。** 加進它的 `.env`：

```bash
OPENTEDDY_FLEET_TOKEN=<步驟 1 的 token>
OPENTEDDY_FLEET_ROLE=orchestrator
OPENTEDDY_FLEET_PORT=8770
```

**步驟 3 — 每個 worker 節點。** 加進它的 `.env`：

```bash
OPENTEDDY_FLEET_TOKEN=<一樣的 token>
OPENTEDDY_FLEET_ROLE=worker
OPENTEDDY_FLEET_CENTRAL=ws://<中控主機名或IP>:8770
OPENTEDDY_FLEET_NODE_ID=dgx-02            # 這台的名字
OPENTEDDY_FLEET_NODE_ROLE=finance         # 職責：finance / secops / …
# 可選 — 主動監控：
OPENTEDDY_FLEET_WATCH_ENABLED=1
OPENTEDDY_FLEET_WATCH_PROMPT=檢查 /data/finance 最近一小時是否有異常大額付款
```

有附完整註解的範本可直接用：
`cat fleet/env.orchestrator.example >> .env`（中控）/
`cat fleet/env.worker.example >> .env`（worker）。

**步驟 4 — 各節點重啟，然後在中控測試：**

```bash
# 列出連入的節點
curl -s http://localhost:8000/fleet/nodes | python3 -m json.tool

# 派任務（自動挑一台空閒的 worker）
curl -s -X POST http://localhost:8000/fleet/dispatch \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"auto","goal":"整理今日 GitHub trending top 5","mode":"code"}' \
  | python3 -m json.tool

# worker watcher 主動推的警報
curl -s http://localhost:8000/fleet/alerts | python3 -m json.tool
```

或直接開**網頁中控台** `http://<中控>:8000/fleet` —— 三個分頁：**Workers**（即時狀態）、**Playground**（輸入任務 → 自動挑空閒節點）、**Alerts**（主動異常回報）。

完整指南：[`fleet/README.md`](fleet/README.md) ·
設計文件：[`docs/fleet-architecture.md`](docs/fleet-architecture.md)。

## 平台支援

| 作業系統 | 狀態 | 備註 |
|----------|------|------|
| **macOS**（Apple Silicon） | ✅ 原生桌面版 `.dmg`（已簽章＋公證）+ OSS web | 主要開發環境 |
| **Linux** (x86_64) | ✅ 原生桌面版 `.AppImage` / `.deb` *(v1.0.3 新增)* + OSS web | 透過 GitHub Actions Ubuntu 22.04 runner 自動建置，AppImage 適用任何 glibc 2.34+ 的發行版 |
| **Windows（原生）** | ⚠️ 部分支援，建議改用 WSL2 | 見下方注意事項。原生桌面版安裝程式尚在規劃中 |
| **Windows（WSL2）** | ✅ 完全支援（OSS web） | 等同於 Linux，Windows 使用者推薦此路徑 |

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

## ☕ 請我喝咖啡贊助

OpenTeddy 是一人 side project。Server 費、雲端 LLM API 測試、桌面版簽章
公證流程、加上週末持續寫新工具的時間，每個月都是實際成本。如果這個專案
幫你節省時間，歡迎贊助一杯：

[**☕ Buy Me a Coffee →**](https://openteddy-app.lemonsqueezy.com/checkout/buy/103ae6c2-36cf-48e1-aefc-71faca140657)

沒有分級訂閱、沒有功能鎖、沒有「premium」 — repo 內所有功能永遠 MIT
免費。贊助只是讓我能多花一點時間繼續加工具、修 planner edge case、
寫沒有人會寫的文件。

## 跑大模型（DGX Spark / 35B 等級）的人請注意

**系統提示詞是固定文字，KV cache 可以重複使用。** 以前每次呼叫模型，系統提示詞開頭都放了精確到秒的時間，discovery memo 也接在系統提示詞後面、工具定義前面，所以沒有兩次呼叫的開頭一樣，Ollama 每次都要把整份提示詞（系統文字加約 60 個工具定義，大約 1.2 萬 token）從頭讀一遍。現在時間改放在使用者訊息裡（只到分鐘），memo 放在對話最後。在 GX10（DGX Spark 等級）+ qwen3.8:27b 上實測：每個新子任務讀提示詞的時間 **14.1 秒 → 1.5 秒**，有 memo 的下一輪 1.6 秒 → 0.4 秒。用 `scripts/prompt_cache_bench.py --url http://<ollama>:11434 --model <執行模型>` 可以在你的機器上量。每次地端呼叫也會在 log 寫一行時間花在哪（`LLM executor r0 qwen3.8:27b: load 0 ms · prefill 11941 tok 1.5 s · decode 88 tok 6.1 s`），用量紀錄另存 `load_ms` / `prefill_ms`。

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

OpenTeddy 本體：**MIT**。

repo 內附帶的第三方內容：

| 附帶檔案 | 上游來源 | 上游授權 |
|---|---|---|
| `cyber_skills/index.json`（內建 755 個專家工作流程的索引） | [mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)（754 條）+ [mvanhorn/last30days-skill](https://github.com/mvanhorn/last30days-skill)（1 條） | Apache 2.0 + MIT |
| `tools/doc_to_markdown.py`（包裝 [microsoft/markitdown](https://github.com/microsoft/markitdown)） | 上游 PyPI 套件 | Apache 2.0 |

`cyber_skills/index.json` 是**衍生作品（derivative work）** — 詳見
`cyber_skills/README.md`。索引中的每一筆都有 `source_repo` 與
`upstream_url` 欄位，可以追溯到原始 repo。完整的授權條款與
NOTICE 檔（若有）請看上面連結的上游 repo。
