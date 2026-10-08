from src.preprocessing import clean_text


def test_lowercase():
    result = clean_text("HELLO WORLD")

    assert result == "hello world"


def test_remove_special_characters():
    result = clean_text("Hello!!! How are you???")

    assert result == "hello how are you"


def test_remove_url():
    result = clean_text("Visit http://example.com now")

    assert result == "visit now"


def test_remove_extra_spaces():
    result = clean_text("Hello     world")

    assert result == "hello world"