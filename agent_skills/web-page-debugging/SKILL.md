---
name: web-page-debugging
description: A web page, HTML report or browser game shows the wrong thing — something missing, blank, covered, too faint, out of place, not moving, or not reacting to keys or clicks (網頁/遊戲畫面看不到、空白、跑版、沒反應). How to see the problem with page_look and the usual causes.
---
# 網頁 / 遊戲畫面除錯

## 先看到問題
```
page_look(target="game.html", actions=["click 開始遊戲", "wait 1000"], question="看得到主角嗎？")
```
- `target`：工作目錄裡的 html 檔，或 localhost / 區網的網址
- `actions`：照使用者的操作 —— `click <按鈕文字或 CSS 選擇器>`、`press ArrowRight`、`type 文字`、`wait 毫秒`
- 讀回報：
  - `covered`：哪個元素被誰蓋住（最常見的「看不到」原因）
  - `errors`：JS 錯誤、載入失敗的檔案
  - `seen`：看圖模型描述畫面上真的有什麼

## 常見原因（依現象找）
- **整個畫面只剩背景或特效，內容都不見** → 有全畫面元素蓋在上面：`position: fixed/absolute` 的背景特效 canvas、遮罩、overlay，有不透明背景或 z-index 比內容高。
  修法：特效層 `z-index: -1` 且背景透明；或內容 `position: relative; z-index: 1`。
- **樣式選擇器太廣**：`canvas {…}`、`div {…}` 這種標籤選擇器會套到所有同類元素 —— 例如給遊戲畫布的底色，也塗到了背景特效 canvas 上，把遊戲蓋住。改成 `#id` 或 class。
- **canvas 上的東西太淡或看不見**：`ctx.fillStyle` / `ctx.globalAlpha` 沿用了前面的半透明值（例如畫格線用 `rgba(255,255,255,0.03)`，之後畫 emoji、文字沒重設；emoji 也會吃這個透明度）。畫之前重設 `fillStyle` 和 `globalAlpha = 1`。
- **東西畫在畫面外或大小是 0**：座標算錯；canvas 的 width/height 屬性跟 CSS 尺寸不一致；元素 `display: none`。
- **按了沒反應**：先看 `errors` 有沒有 JS 錯誤；事件綁錯元素；遊戲迴圈（setInterval / requestAnimationFrame）沒啟動或被停掉。
- **在使用者那邊才壞**：檔案或圖片路徑錯、外部資源載入失敗（`errors` 會列出失敗的請求）。

## 修完一定要再看
直接改 html 檔本身，再用**同樣的 actions** 跑一次 `page_look`：原本看不到的東西出現了、`covered` 沒有它、`errors` 沒有新錯誤，才算修好。
