
import string
import re

from spacy.lang.en.stop_words import STOP_WORDS
# from nltk.corpus import stopwords
# STOP_WORDS = set(stopwords.words("english"))

# for sentiment analysis we need words that change meaning to the opposite
STOP_WORDS.discard('not') 
STOP_WORDS.discard('never') 
STOP_WORDS.discard('none') 
STOP_WORDS.discard('nor') 

import spacy

nlp = spacy.blank("en")

REGX_USERNAME = r"@[A-Za-z0-9$-_@.&+]+"
REGX_URL = r"https?://[A-Za-z0-9./]+"


def decontract(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"1st", "first", phrase)
    phrase = re.sub(r"2nd", "second", phrase)
    phrase = re.sub(r"3rd", "third", phrase)

    return phrase


def preprocessing(text):
  text = text.lower()
  
  # extra html line breaks
  text = re.sub('<br />', ' ', text)

  text = decontract(text)

  text = re.sub(REGX_USERNAME, ' ', text)
  text = re.sub(REGX_URL, ' ', text)
  
  emojis = {
      ':)': 'emotion positive',
      ':(': 'emotion negative'
  }
  
  for e in emojis:
    text = text.replace(e, emojis[e])
  
  tokens = [token.text for token in nlp(text)]
  
  # print('\n-> test:', " ".join(tokens))
  # print(STOP_WORDS)

  tokens = [t for t in tokens if 
              t not in STOP_WORDS and 
              t not in string.punctuation and 
              len(t) > 2
              ]
  
  tokens = [t for t in tokens if not t.isdigit()]
    
  return " ".join(tokens)


if __name__ == '__main__':
    text = 'This is just a test text :) 1234.<br />Don\'t pay attention to it.'
    print(f'"{text}"\n -> "{preprocessing(text)}"')
