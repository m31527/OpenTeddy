from slugify import slugify


def test_basic():
    assert slugify('Hello World') == 'hello-world'


def test_punctuation():
    assert slugify('  OpenTeddy: Local AI, Agent!  ') == 'openteddy-local-ai-agent'


def test_repeats():
    assert slugify('a -- b__c') == 'a-b-c'
