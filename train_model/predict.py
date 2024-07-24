
from preprocess import preprocessing

SENTIMENT_THRESHOLD = 0.5 # for POS
SENTIMENT_THRESHOLD2 = 0.5 # for NEG

MIN_LENGTH = 5

def spacy_test_text(nlp, text, verbose=False):
    if len(text)<MIN_LENGTH:
        print(f'\n"{text}": text is too short, must be at least {MIN_LENGTH}')
        return text, {'POS': 0, 'NEG': 0} # , 'SENTIMENT':0
    doc = nlp(preprocessing(text))
    if verbose:
        print(f'"{text}"\n -> "{doc}"\n {doc.cats}')
    return doc, doc.cats


def spacy_test_list(nlp, text_list, verbose=False):
    results = []
    for text in text_list:
        doc, doc_cats = spacy_test_text(nlp, text, verbose)
        results.append((doc, doc_cats))
    return results


def spacy_get_sentiment_preprocess(text, nlp, verbose=False):
    doc, doc_cats = spacy_test_text(nlp, text, verbose)
    return 1 if doc_cats['POS']>SENTIMENT_THRESHOLD else 0


def spacy_get_sentiment(text, nlp, verbose=False):
    # version without preprocessing
    MIN_LENGTH = 5
    if len(text)<MIN_LENGTH:
        print(f'\n"{text}": text is too short, must be at least {MIN_LENGTH}')
        return 0
    doc = nlp(text)
    if verbose:
        print(f'{text}\n{doc}\n {doc.cats}')
    return 1 if doc.cats['POS']>SENTIMENT_THRESHOLD else 0


if __name__ == '__main__':
    MODEL_DIR = './model/model-best/model.spacy/'

    import spacy
    nlp = spacy.load(f"{MODEL_DIR}")
    print(f'\nTesting model {MODEL_DIR}')
    text = 'I liked that book, it was fun'
    spacy_test_text(nlp, text, verbose=True)
