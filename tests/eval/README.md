# 交付壓測（delivery eval）

量 OpenTeddy 是不是**真的把事情做完**，不只是回報「完成」。20 題，每題都有自動檢查：用模型沒看過的測試跑它寫的程式、拿數字和標準答案比對、確認檔案真的產生，也測「不該做的不要做」（訂不到的機票、道謝、刪檔案）。

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
- **改題目**：輸入檔和標準答案由 `make_fixtures.py` 產生，改了要一起重跑；檢查本身的測試是 `tests/test_eval_checks.py`。
