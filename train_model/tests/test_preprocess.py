from preprocess import (
    decontract,
    preprocessing,
)

def test_decontract():
    text = "I didn't like reading the book"
    actual_result = decontract(text)
    expected_result = "I did not like reading the book"
    assert actual_result == expected_result


def test_preprocessing():
    text = "I didn't like reading the book"
    actual_result = preprocessing(text)
    expected_result = "not like reading book"
    assert actual_result == expected_result