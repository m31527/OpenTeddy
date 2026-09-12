# Shopify Admin API（Agent 專用說明）

## 唯一的呼叫方式

所有請求都是 **POST** 到 GraphQL 端點，用 `http_post`：

```
url:     https://{{CRED:shopify_store}}/admin/api/2026-07/graphql.json
headers: {"X-Shopify-Access-Token": "{{CRED:shopify_token}}", "Content-Type": "application/json"}
body:    {"query": "<GraphQL 查詢字串>", "variables": {...}}
```

- 憑證一律寫 `{{CRED:shopify_token}}` / `{{CRED:shopify_store}}`，不要猜測、不要用 python_exec 自己連線
- 回傳是 JSON；先看 `errors`（GraphQL 錯誤）再看 `data`
- 一次 GraphQL 查詢通常就能拿到一天的資料；不要逐筆查

## 1. 每日訂單與歸因（主要資料來源）

`orders` 連線可帶 `query` 篩選；`customerJourneySummary` 是廣告歸因的依據（來源、UTM）。

```graphql
query DailyOrders($q: String!, $after: String) {
  orders(first: 100, query: $q, after: $after, sortKey: CREATED_AT) {
    pageInfo { hasNextPage endCursor }
    edges { node {
      id name createdAt
      displayFinancialStatus displayFulfillmentStatus
      totalPriceSet { shopMoney { amount currencyCode } }
      currentTotalPriceSet { shopMoney { amount currencyCode } }
      customerJourneySummary {
        momentsCount customerOrderIndex
        firstVisit { source sourceType referrerUrl landingPage
                     utmParameters { source medium campaign content term } }
        lastVisit  { source sourceType referrerUrl landingPage
                     utmParameters { source medium campaign content term } }
      }
      lineItems(first: 20) { edges { node { title quantity
        originalTotalSet { shopMoney { amount } } } } }
    } }
  }
}
```

variables 範例（昨天，已付款）：
```json
{"q": "created_at:>='2026-09-11T00:00:00+08:00' created_at:<'2026-09-12T00:00:00+08:00' financial_status:paid"}
```

規則：
- `hasNextPage` 為 true 就帶 `after: endCursor` 再查，直到 false
- 廣告歸因看 `lastVisit.utmParameters.campaign`（最後點擊）；`firstVisit` 是首次觸及
- 沒有 `customerJourneySummary`（null）的訂單歸為「直接/未知」，不要臆測來源
- 預設只能查最近 60 天；更早的需要 `read_all_orders`

## 2. 彙總數字：ShopifyQL（次要，做趨勢用）

```graphql
query { shopifyqlQuery(query: "FROM sales SHOW total_sales GROUP BY day SINCE -7d ORDER BY day") {
  tableData { columns { name dataType displayName } rows }
  parseErrors
} }
```

- 需要 `read_reports` 權限
- **先看 `parseErrors`**；不是空陣列就代表語法錯，改寫再試，不要拿空結果當 0
- 已驗證可用：`FROM sales SHOW total_sales GROUP BY month SINCE -3m ORDER BY month`
- 其他指標名稱以 parseErrors 的回饋為準，不要臆造

## 3. 商品與庫存（唯讀）

```graphql
query { products(first: 50, query: "status:active") { edges { node {
  id title status totalInventory
  variants(first: 10) { edges { node { id title price inventoryQuantity } } }
} } } }
```

## 4. 寫入：部落格文章「草稿」（素材 agent 才有此權限）

只建立草稿，**絕不發布**（`isPublished: false`），發布由人在後台按。

```graphql
mutation CreateDraft($article: ArticleCreateInput!) {
  articleCreate(article: $article) {
    article { id title isPublished }
    userErrors { field message }
  }
}
```
variables：
```json
{"article": {"blogId": "gid://shopify/Blog/<id>", "title": "…", "body": "<p>…</p>", "isPublished": false, "author": {"name": "OpenTeddy"}}}
```
先用 `query { blogs(first: 5) { edges { node { id title } } } }` 找 blogId。
若回傳 `Field 'articleCreate' doesn't exist`，改用 REST：
`POST https://{{CRED:shopify_store}}/admin/api/2026-07/blogs/<blog_id>/articles.json`，body `{"article": {"title": "…", "body_html": "…", "published": false}}`。

## 判斷成功／失敗

- 成功：HTTP 200 且 `errors` 不存在、`userErrors` 為空
- 失敗：HTTP 401/403 = token 或權限問題；`errors[].message` 含 `access denied` = 缺少該 scope（read_orders / read_reports / write_content）；**要照實回報缺哪個權限，不要略過**
- 429 / `THROTTLED`：等 2 秒重試一次，仍失敗就回報

## 規則

- 所有內部營運數字**只從這裡查**，不要用 web_search
- 報告裡引用的數字要能對回查詢結果；查詢區間內 0 筆時，多查一次最早/最新訂單日期並寫明「資料最新到 X 月 X 日」
- 端點只有上面這些，不要臆造其他 REST 路徑
