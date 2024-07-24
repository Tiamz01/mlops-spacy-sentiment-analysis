import spacy

from predict import (
    spacy_test_list,
    # spacy_test_text,
)
from predict import SENTIMENT_THRESHOLD, SENTIMENT_THRESHOLD2

def test_spacy_test_text():
    MODEL_DIR = './model/model-best/model.spacy/'
    nlp = spacy.load(f"{MODEL_DIR}")
    print(f'\nTesting model {MODEL_DIR}')

    texts = [
        # "?!",
        "Text is too short. I didn't like it",
        "I didn't like reading the book",
        "I didn't like the book",
        "I liked the book",
        "Just don't buy it",
        # 'I do not dislike cabin cruisers',
        # 'Disliking watercraft is not really my thing',
        'Sometimes I really hate RIBs.',
        "I'd really truly love going out in this weather! ",
        'The movie is surprising, with plenty of unsettling plot twists',
        # 'You should see their decadent dessert menu.',
        # 'I love my mobile but would not recommend it to any of my colleagues',
        # "I dislike old cabin cruisers",
    ]

    expected_result = [
        # ("?!", {"POS": 0, "NEG": 0}),
        ("text short not like", {"POS": 0, "NEG": 1}),
        ("not like reading book", {"POS": 0, "NEG": 1}),
        ("not like book", {"POS": 0, "NEG": 1}),
        ("liked book", {"POS": 1, "NEG": 0}),
        ("not buy", {"POS": 0, "NEG": 1}),
        # ('not dislike cabin cruisers',{'POS':0, 'NEG':0}),
        # ('disliking watercraft not thing',{'POS':0, 'NEG':0}),
        ('hate ribs',{'POS':0, 'NEG':1}),
        ("truly love going weather", {"POS": 1, "NEG": 0}),
        ('movie surprising plenty unsettling plot twists',{'POS':1, 'NEG':0}),
        # ('decadent dessert menu',{'POS':0, 'NEG':0}),
        # ('love mobile not recommend colleagues',{'POS':1, 'NEG':0}),
        # ("dislike old cabin cruisers", {"POS": 0, "NEG": 1}),
    ]

    actual_result = spacy_test_list(nlp, texts, verbose=False)

    assert len(actual_result) == len(expected_result)

    for i in range(len(actual_result)):
        assert str(actual_result[i][0]) == expected_result[i][0]

        # POS/NEG values may vary, but must be higher than THRESHOLD 
        if expected_result[i][1].get("POS") == 1:
            assert actual_result[i][1].get("POS") > SENTIMENT_THRESHOLD
        elif expected_result[i][1].get("NEG") == 1:
            assert actual_result[i][1].get("NEG") > SENTIMENT_THRESHOLD2
        else: # both 0 when too short
            assert actual_result[i][1].get("NEG") == 0
            assert actual_result[i][1].get("NEG") == 0
