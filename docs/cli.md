# OpenTeddy CLI 速查

`openteddy` 是 OpenTeddy Runtime 的終端機入口。它只負責送出任務、即時顯示進度、在終端機回答核准提示;規劃與執行都在 runtime(背景服務)裡進行,所以**關掉網頁也能完成任務**。

## 安裝與連線

```bash
cd ~/OpenTeddy
mkdir -p ~/.local/bin && ln -sf "$(pwd)/openteddy" ~/.local/bin/openteddy
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

openteddy health        # ✓ runtime up 就是連上了
```

預設連 `http://127.0.0.1:8000`。要連另一台機器(例如從 Mac 連 DGX):

```bash
export OPENTEDDY_URL=http://<DGX 的 Tailscale IP>:8000     # 寫進 ~/.zshrc 或 ~/.bashrc
```

## 最常用的五個指令

```bash
openteddy run "查今天新增訂單數，跟昨天比" --agent EasyBuy    # 用某個 agent 做事
openteddy run "修好 tests/test_api.py 失敗的測試" --dir .      # 在目前的專案目錄裡做事
openteddy task list                                            # 最近的任務
openteddy task status a1b2c3d4 --follow                        # 接回一個還在跑的任務
openteddy run "今天有沒有需要注意的"                            # 所有排程的最新狀況
```

## run:送出任務

```bash
openteddy run "<要做的事>" [參數]
```

| 參數 | 作用 |
|---|---|
| `--agent 名稱` | 用某個 agent 執行(帶上它的人設、資料庫、憑證、工具範圍)。名稱打其中一段就行,例如 `--agent EasyBuy` |
| `--dir .` | 在這個專案目錄裡工作(開一個 code 模式 session)。路徑必須存在於**跑 OpenTeddy 的那台機器** |
| `--session ID` | 接續既有的 session |
| `--mode chat/code/analytic` | 指定模式 |
| `--local-only` | 這次任務絕不使用雲端模型 |
| `--unattended` | 不跳核准提示直接執行。**只有設了工具範圍的 agent 可以用** |
| `--yes` | 所有核准提示都自動回答 yes |
| `--no-follow` | 送出後只印任務 ID 就返回,不等結果 |

執行時會即時顯示計畫、每次工具呼叫、產出的檔案(📎)和最後結果。遇到高風險動作會停下來問:

```
⚠ shell_exec_write needs approval:
    {"command": "git commit -m ..."}
  approve? [y/N]
```

用說的就能建排程,不會立刻執行:

```bash
openteddy run "每天早上 8 點查昨日營收，跟前 7 天平均比，偏差超過 15% 才通知我" --agent EasyBuy
```

## task:管理任務

```bash
openteddy task list [--limit 20] [--session ID]
openteddy task status ID [--follow] [--yes]
openteddy task approve ID       # 核准這個任務所有待核准的動作
openteddy task reject ID
openteddy task cancel ID
```

ID 只要打前幾碼(例如 `a1b2c3d4`)。

## agent:查看 agent、設定工具範圍

```bash
openteddy agent list                                       # 每個 agent 的模式、資料庫、工具範圍
openteddy agent scope EasyBuy                              # 顯示目前的工具範圍
openteddy agent scope EasyBuy db_query render_chart_report # 設定工具範圍（權限邊界）
openteddy agent scope EasyBuy --clear                      # 清除範圍（= 所有工具）
openteddy tools                                            # 所有工具名稱與風險等級
```

工具範圍是 `--unattended` 和「排程自動核准」的前提。

## schedule:排程

```bash
openteddy schedule list                      # 🔔 = 上次有通知 · = 檢查過、沒事
openteddy schedule run ID                    # 立刻跑一次，不用等時間到
openteddy schedule on ID | off ID
openteddy schedule notify ID "偏差超過 20%"   # 設定通知條件
openteddy schedule notify ID                 # 清除條件（= 每次都通知）
openteddy schedule delete ID
```

## skill:技能

```bash
openteddy skill list
openteddy skill run 技能名稱 --input '{"path": "~/data/q3.csv"}'   # 直接執行，不經規劃
```

## 查看系統狀態

```bash
openteddy health          # runtime 是否在線、版本
openteddy models          # 規劃 / 執行 / 語音模型、雲端供應商
openteddy decisions       # 決策引擎（Laya 影子模式）每種判斷的一致率
```

## 背景服務與更新

```bash
openteddy service install --host 0.0.0.0     # 裝成開機自動啟動的服務
sudo loginctl enable-linger $USER            # Linux 無人登入時也要啟動：只需做一次
openteddy service status | logs | restart | stop | start | uninstall

openteddy update                             # git pull → 需要時裝依賴 → 重啟服務 → 健康檢查
```

用 `--system` 裝成系統服務的話,這些指令也要加 `--system`。

## 小提醒

- `--json` 和 `--url` 要放在子指令**前面**:`openteddy --json task list`、`openteddy --url http://x:8000 health`
- 在 `run` 或 `--follow` 中按 `Ctrl+C` 只是停止顯示,**任務會繼續跑**;要停止請用 `task cancel`
- 結束代碼:`0` 完成、`1` 失敗、`2` 已升級給雲端,可直接用在 shell 腳本裡
- 名稱有空格或括號時要加引號:`--agent "EasyBuy 數據 (UAT)"`;或只打其中一段 `--agent EasyBuy`

## 常見問題

| 看到 | 原因與解法 |
|---|---|
| `Runtime not reachable` | 服務沒啟動:`openteddy service start` 或 `./run.sh` |
| `HTTP 400 … needs an agent with a declared tool scope` | 用了 `--unattended` 但 agent 沒有工具範圍:先 `openteddy agent scope …` |
| `workspace_dir not found on the runtime's machine` | `--dir` 的路徑在跑 OpenTeddy 的那台機器上不存在 |
| `Ollama 佇列已滿` | 在伺服器執行 `sudo systemctl restart ollama`,再用 `ollama ps` 確認模型載得進去 |
| `No agent matches` / `is ambiguous` | 用 `openteddy agent list` 看名稱,打更完整的一段或 ID 前幾碼 |
