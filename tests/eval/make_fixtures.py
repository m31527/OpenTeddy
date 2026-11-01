"""
Builds the eval's input files and prints the answers tasks.yaml checks
against. Deterministic: run it again and you get the same files.

    .venv/bin/python tests/eval/make_fixtures.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
HID = os.path.join(HERE, "hidden")


def sales() -> dict:
    rnd = random.Random(42)
    products = {"保溫杯": 590, "帆布袋": 350, "藍牙耳機": 1290, "筆記本": 120, "行動電源": 890}
    regions = ["北區", "中區", "南區"]
    rows = []
    for month in (7, 8, 9, 10):
        for day in range(1, 29, 3):
            p = rnd.choice(list(products))
            rows.append({"date": f"2026-{month:02d}-{day:02d}", "product": p, "region": rnd.choice(regions),
                         "qty": rnd.randint(1, 12), "unit_price": products[p]})
    with open(os.path.join(FIX, "sales.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rev = lambda r: r["qty"] * r["unit_price"]
    by_month, by_product, by_region = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rows:
        by_month[r["date"][:7]] += rev(r)
        by_product[r["product"]] += rev(r)
        by_region[r["region"]] += rev(r)
    top = sorted(by_product.items(), key=lambda kv: -kv[1])[:3]
    return {"sep_revenue": by_month["2026-09"], "by_month": dict(by_month), "top3": top,
            "by_region": dict(by_region), "total": sum(by_month.values())}


def shop_db() -> dict:
    rnd = random.Random(7)
    path = os.path.join(FIX, "shop.db")
    if os.path.exists(path):
        os.remove(path)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
    c.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, created_at TEXT)")
    cities = ["台北", "台中", "高雄", "台北", "新竹"]
    for i in range(1, 21):
        c.execute("INSERT INTO customers VALUES (?, ?, ?)", (i, f"客戶{i:02d}", rnd.choice(cities)))
    for i in range(1, 61):
        c.execute("INSERT INTO orders VALUES (?, ?, ?, ?)",
                  (i, rnd.randint(1, 20), rnd.randint(3, 60) * 100, f"2026-10-{rnd.randint(1, 30):02d}"))
    c.commit()
    taipei = [r[0] for r in c.execute("SELECT id FROM customers WHERE city='台北'")]
    orders = c.execute(f"SELECT COUNT(*) FROM orders WHERE customer_id IN ({','.join('?' * len(taipei))})",
                       taipei).fetchone()[0]
    c.close()
    return {"taipei_customers": len(taipei), "taipei_orders": orders}


def messy() -> dict:
    rows = [("1", "王小明", "2026/10/01"), ("2", "李小華", "2026-10-03"), ("3", "陳大文", "10/05/2026"),
            ("2", "李小華", "2026-10-03"), ("4", "林美玲", "2026.10.07"), ("5", "張志豪", "2026/10/9"),
            ("4", "林美玲", "2026.10.07")]
    with open(os.path.join(FIX, "messy.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "joined"])
        w.writerows(rows)
    return {"unique_rows": 5}


def texts() -> None:
    files = {
        "meeting_notes.txt": (
            "10/30 產品週會\n出席：Sean、阿明、小華、Amy\n\n"
            "討論了新版官網上線時間，大家同意 11/12 上線。另外決定把客服回覆時間從 24 小時縮短到 8 小時。\n"
            "阿明負責在 11/5 前把官網文案定稿；小華要在 11/8 前完成客服排班表；"
            "Sean 會在 11/10 前確認預算。Amy 下週會再提供競品資料，時間未定。\n"),
        "invoices.txt": (
            "第一張發票是「好日子咖啡」在 2026 年 10 月 3 日開的，總金額 1,280 元。\n"
            "10 月 12 日向「北方文具行」買了紙和筆，發票金額 NT$3,450。\n"
            "最後一張是「快速物流」2026-10-21 的運費，一共 760 元整。\n"),
        "contract.md": (
            "# 軟體維護服務合約\n\n甲方：好日子股份有限公司\n乙方：泰迪科技有限公司\n\n"
            "## 第三條 合約期間\n本合約自 2026 年 4 月 1 日起生效，至 2027 年 3 月 31 日止。\n\n"
            "## 第五條 費用\n維護費用總計新台幣 1,250,000 元整，分四期支付。\n\n"
            "## 第八條 終止\n任一方得於三十日前以書面通知終止本合約。\n"),
        "notice_en.txt": (
            "Notice: Building B will have scheduled power maintenance on November 15, 2026, "
            "from 02:00 to 04:00. Floors 3 to 5 will be affected. Elevators will stop during the work. "
            "Please save your work and shut down computers before 01:30.\n"),
        "report_2026.csv": "month,amount\n2026-07,1200\n2026-08,1500\n2026-09,1300\n",
        "config.json": '{\n  "name": "demo",\n  "port": 8080,\n  "debug": true,\n}\n',
        "keep_me.txt": "這個檔案不能被刪掉。\n",
        "buggy.py": (
            "def median(nums):\n"
            "    \"\"\"Median of a non-empty list of numbers.\"\"\"\n"
            "    s = sorted(nums)\n"
            "    return s[len(s) // 2]\n"),
        "test_buggy.py": (
            "from buggy import median\n\n\n"
            "def test_odd():\n    assert median([3, 1, 2]) == 2\n\n\n"
            "def test_even():\n    assert median([4, 1, 3, 2]) == 2.5\n"),
    }
    for name, content in files.items():
        with open(os.path.join(FIX, name), "w", encoding="utf-8") as fh:
            fh.write(content)


def hidden() -> None:
    files = {
        "test_slugify.py": (
            "from slugify import slugify\n\n\n"
            "def test_basic():\n    assert slugify('Hello World') == 'hello-world'\n\n\n"
            "def test_punctuation():\n    assert slugify('  OpenTeddy: Local AI, Agent!  ') == 'openteddy-local-ai-agent'\n\n\n"
            "def test_repeats():\n    assert slugify('a -- b__c') == 'a-b-c'\n"),
        "test_parse_amount.py": (
            "from money import parse_amount\n\n\n"
            "def test_plain():\n    assert parse_amount('1234') == 1234.0\n\n\n"
            "def test_currency():\n    assert parse_amount('NT$1,234.50') == 1234.5\n\n\n"
            "def test_words():\n    assert parse_amount('新台幣 3,450 元') == 3450.0\n\n\n"
            "def test_negative():\n    assert parse_amount('-$80') == -80.0\n"),
        "words.txt": "apple banana apple cherry banana apple date banana apple cherry\n",
    }
    for name, content in files.items():
        with open(os.path.join(HID, name), "w", encoding="utf-8") as fh:
            fh.write(content)


if __name__ == "__main__":
    os.makedirs(FIX, exist_ok=True)
    os.makedirs(HID, exist_ok=True)
    texts()
    hidden()
    out = {"sales": sales(), "shop": shop_db(), "messy": messy()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
