# Meta Marketing API（Agent 專用說明）

## 唯一的呼叫方式

基底：`https://graph.facebook.com/v25.0`（以你 App 可用的版本為準；第一次先呼叫一次 insights 確認版本沒被拒）

- 讀取用 `http_get`，寫入用 `http_post`
- 憑證用 query 參數：`access_token={{CRED:meta_token}}`，一律寫佔位符，不要猜測
- 廣告帳號 ID 用 `{{CRED:meta_ad_account}}`（格式 `act_1234567890`）
- 回傳 JSON；錯誤長這樣：`{"error": {"message": "...", "code": 190}}`
  - code 190 = token 失效；code 100 = 參數錯；code 17/32/613 = 被限流 → 等 60 秒再試一次

## 1. 成效讀取（分析師的主要資料來源）

昨天、每個 campaign 一列：

```
GET https://graph.facebook.com/v25.0/{{CRED:meta_ad_account}}/insights
    ?level=campaign
    &date_preset=yesterday
    &time_increment=1
    &fields=campaign_id,campaign_name,spend,impressions,clicks,ctr,cpc,cpm,reach,frequency,actions,action_values,cost_per_action_type,purchase_roas
    &limit=200
    &access_token={{CRED:meta_token}}
```

- `level` 可為 `account` / `campaign` / `adset` / `ad`（要看素材成效用 `ad`，加 `ad_id,ad_name,adset_id,adset_name`）
- 指定區間：`time_range={"since":"2026-09-01","until":"2026-09-11"}` 取代 `date_preset`
- 分頁：回傳有 `paging.next` 就繼續 GET 那個 URL
- 數字都是**字串**，要轉成數值再算

**購買數與購買金額**在 `actions` / `action_values` 陣列裡，形狀是 `[{"action_type": "...", "value": "12"}]`：
- 購買次數：優先取 `action_type == "omni_purchase"`；沒有才取 `"purchase"`；都沒有再取 `"offsite_conversion.fb_pixel_purchase"`。**只取一種，不要相加**（它們是重疊的）
- 購買金額：同樣規則套用在 `action_values`
- `purchase_roas` 若存在是 `[{"action_type": "omni_purchase", "value": "3.21"}]`
- CPA = spend ÷ 購買次數；購買為 0 時 CPA 記為 null，不要除以零、不要寫「無限大」

## 2. 廣告物件狀態（唯讀）

```
GET https://graph.facebook.com/v25.0/{{CRED:meta_ad_account}}/campaigns?fields=id,name,status,effective_status,daily_budget,lifetime_budget,objective&limit=200&access_token={{CRED:meta_token}}
GET https://graph.facebook.com/v25.0/{{CRED:meta_ad_account}}/adsets?fields=id,name,status,effective_status,daily_budget,campaign_id&limit=200&access_token={{CRED:meta_token}}
GET https://graph.facebook.com/v25.0/{{CRED:meta_ad_account}}/ads?fields=id,name,status,effective_status,adset_id,creative{id,name}&limit=200&access_token={{CRED:meta_token}}
```
預算單位是**最小貨幣單位**（美金帳號：分；5000 = $50.00）。

## 3. 寫入（投手 agent；受花錢護欄限制）

OpenTeddy 的伺服器會對這些呼叫強制執行規則，**不論是否經過核准**：

| 動作 | 結果 |
|---|---|
| 暫停：`POST /<id>` form `status=PAUSED` | ✅ 永遠允許 |
| 啟用：`status=ACTIVE` | 🚫 拒絕（由人在 Ads Manager 開啟） |
| 建立 campaign / adset / ad | ✅ 只在 `status=PAUSED` 時允許 |
| 設定 `daily_budget` / `lifetime_budget` | ✅ 只在 ≤ 設定的上限時允許；每天最多 3 次 |
| 建立 adcreative（素材） | ✅ 允許（它本身不花錢） |

被拒絕時工具會回 `🚫 Spend guard: …`。**不要換個寫法重試**，把它列進「需要人工執行的建議」。

範例：
```
POST https://graph.facebook.com/v25.0/<adset_id>
form: {"daily_budget": "5000", "access_token": "{{CRED:meta_token}}"}

POST https://graph.facebook.com/v25.0/<campaign_id>
form: {"status": "PAUSED", "access_token": "{{CRED:meta_token}}"}

POST https://graph.facebook.com/v25.0/{{CRED:meta_ad_account}}/campaigns
form: {"name": "…", "objective": "OUTCOME_SALES", "status": "PAUSED", "special_ad_categories": "[]", "access_token": "{{CRED:meta_token}}"}
```

## 判斷成功／失敗

- 成功：HTTP 200 且沒有 `error`；寫入成功會回 `{"success": true}` 或 `{"id": "…"}`
- 失敗要**照實回報 `error.message`**，不要改寫成「已完成」
- insights 回 `data: []` 代表該區間沒有投放，不是錯誤；報告要寫明

## 規則

- 每次任務**只拉一次** insights（同一區間），把結果寫進帳本再分析；不要對同一區間反覆呼叫
- 所有建議動作都要附：物件 ID、目前數值、建議數值、依據的指標（例如「CPA 480 > 目標 300，曝光 2,400 已足夠判斷」）
- 端點只有上面這些，不要臆造
