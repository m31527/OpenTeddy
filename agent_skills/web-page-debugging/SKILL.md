---
name: web-page-debugging
description: A web page, HTML report or browser game shows the wrong thing — something missing, blank, covered, too faint, out of place, not moving, or not reacting to keys or clicks (網頁/遊戲畫面看不到、空白、跑版、沒反應). How to see the problem with page_look and the usual causes.
---
# 網頁 / 遊戲畫面除錯

## 規劃：看 → 改 → 再看，放在同一個子任務
一個子任務做完整個循環，例如：
「用 page_look 打開 game.html（actions 照使用者的操作：click 開始遊戲、wait 1000；question：看得到主角嗎？），
讀 html 依回報找原因、用 edit_file 改檔，再用 page_look 確認，看不到就繼續找下一個原因」
不要拆成「看」「分析」「改」「驗證」好幾個子任務 —— 驗證沒過時才能回頭再改。

## page_look 怎麼用
- target：工作目錄裡的 html 檔，或 localhost / 區網網址
- actions：照使用者的操作 —— `click 按鈕文字或 CSS 選擇器`、`press ArrowRight`、`type 文字`、`wait 1000`
- question：使用者在意的那件事，例如「看得到主角嗎？」「圖表有畫出來嗎？」
- 回報：
  - `answer`：看圖模型對 question 的 YES / NO —— **NO 就是還沒好**
  - `covered`：哪個元素被誰蓋住
  - `canvas_drawing`：canvas 上畫得幾乎透明、或畫到畫布外的文字 / 圖（看不見的主角、物品常在這裡）
  - `errors`：JS 錯誤、載入失敗
  - `seen`：畫面上實際有什麼

## 常見原因（依現象找）
- **內容不見，只剩背景或某一層** → 有元素蓋在上面（`covered` 會寫是誰）：`position: fixed/absolute` 的全畫面層（特效、遮罩、對話框底）有不透明背景，或 z-index 比內容高。讓那一層退到後面或變透明，或把內容的 z-index 提高。
- **樣式選擇器太廣**：`canvas {…}`、`div {…}` 這種標籤選擇器會套到所有同類元素，包括本來不該有那個樣式的。改成 `#id` 或 class。
- **canvas 上的東西看不見或很淡**（`canvas_drawing` 會寫 drawn at opacity …）：畫它的時候 `ctx.fillStyle` 或 `ctx.globalAlpha` 還是先前設的半透明值 —— canvas 的繪圖狀態會一直沿用到被改掉為止；文字和 emoji 也吃 fillStyle 的透明度。畫之前把要的顏色和 `globalAlpha = 1` 設好。
- **畫到畫布外或大小是 0**：座標算錯；canvas 的 width/height 屬性跟 CSS 尺寸不一致；元素 `display: none`。
- **按了沒反應**：看 `errors` 的 JS 錯誤；事件綁錯元素；遊戲迴圈（setInterval / requestAnimationFrame）沒啟動。
- **在使用者那邊才壞**：檔案或圖片路徑錯、外部資源載入失敗（`errors` 會列出）。

## 什麼時候才算修好
- 用 `edit_file` 改 html 檔本身（old_text 從 read_file 原樣複製幾行，new_text 是改好的樣子）—— 不要把整份檔案貼在回答裡，那不會存檔。改完用**同樣的 actions** 跑 page_look。
- 不管你在哪一步（就算這步叫「驗證」），看到 NO 就直接用 edit_file 修，再看一次。
- `answer` 是 YES、`seen` 看得到使用者要的東西、`covered` 和 `canvas_drawing` 沒有它、`errors` 沒有新錯誤 —— 才算修好。
- `answer` 是 NO 或 seen 說看不到 → **還沒修好**，繼續找原因。不可以用「動畫可能還沒啟動」「主角可能比較小」這類猜測帶過。
- 一個畫面問題常常不只一個原因：修好一個後還是看不到，就照新的回報找下一個。
