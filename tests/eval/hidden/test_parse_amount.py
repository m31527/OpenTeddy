from money import parse_amount


def test_plain():
    assert parse_amount('1234') == 1234.0


def test_currency():
    assert parse_amount('NT$1,234.50') == 1234.5


def test_words():
    assert parse_amount('新台幣 3,450 元') == 3450.0


def test_negative():
    assert parse_amount('-$80') == -80.0
