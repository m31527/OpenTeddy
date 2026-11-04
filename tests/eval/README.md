# 交付壓測（delivery eval）

量 OpenTeddy 是不是**真的把事情做完**，不只是回報「完成」。31 題（20 題一般 + 11 題網頁），每題都有自動檢查：用模型沒看過的測試跑它寫的程式、拿數字和標準答案比對、確認檔案真的產生，也測「不該做的不要做」（訂不到的機票、道謝、刪檔案）。

```bash
# 在跑 OpenTeddy 的那台機器上執行（要讀每題的工作資料夾和資料庫）
.venv/bin/python tests/eval/run_eval.py                         # 20 題各跑一次
.venv/bin/python tests/eval/run_eval.py --runs 3                # 每題 3 次，看穩定度
.venv/bin/python tests/eval/run_eval.py --only doc_contract,remark_thanks
.venv/bin/python tests/eval/run_eval.py --compare results/A.json results/B.json
```

- **結果**：寫到 `results/<時間>.md` 和 `.json`。
- **主要指標**：實際成功率、**假成功率**（回報完成但檢查沒過）、時間 p50 / p95、地端模型秒數、升級雲端次數與費用、迴圈保護觸發次數。
- **熱度保護**：每題開始前查 GPU 溫度，到 `--max-temp`（預設 80°C）就先等它降到 `--cool-to`（65°C）。長時間滿載曾讓這類機器自己斷電。
- **核准**：一般工具自動核准；刪除這類危險指令一律拒絕。
- **工作資料夾**：每題一個，放在 `agent-workspace/eval/<時間>/`；跑完會刪掉這次產生的對話紀錄（`--keep-sessions` 保留）。
- **網頁題（11 題，`category: web`）**：在無頭 Chromium 裡裝真的 OpenTeddy 擴充功能，配對到壓測專用的身分，對 `web_site.py` 這個測試網站操作：找最便宜的商品、看規格、跨頁加總訂單、摘要開著的分頁、填客服表單、拒絕購買、不填卡號、不理會文章裡藏的指令；另外 3 題要看圖：只有圖示的愛心按鈕、畫布上的遊戲按鈕、畫布上的營收圖表。OpenTeddy 要按送出或購買前會先問；`browser_rig.py` 依題目設定在頁面上真的點「允許」或「不要」，並記錄問了幾次。壓測身分要有伺服器上 `.keys/browser_eval.key`（權限 600）的金鑰才拿得到，別人用不了。只跑網頁題：`--category web`。
- **真實網站題**：`live_tasks.yaml`（例如在 momo 搜尋風扇），結果會隨網站變動，不算在標準壓測裡；要跑時加 `--tasks tests/eval/live_tasks.yaml`。
- **改題目**：輸入檔和標準答案由 `make_fixtures.py` 產生，改了要一起重跑；檢查本身的測試是 `tests/test_eval_checks.py`。
