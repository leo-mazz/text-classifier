import string

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


SYMBOLS = set(" ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"])
IGNORE_CHARS = SYMBOLS | STOP_WORDS

nlp = spacy.load('en_core_web_md')
parser = English()

def clean_text(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

class Glove(TransformerMixin):
    def transform(self, X, **transform_params):
        return np.array([nlp(text).vector for text in X])
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

def tokenize_text(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in IGNORE_CHARS]
    return tokens


def train_model(model, data, spacy_tokens=False, glove=False):
    assert not (spacy_tokens and glove)

    text, labels = data
    classes = labels.unique()

    if glove:
        vectorizer = Glove()
    elif spacy_tokens:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
    else:
        vectorizer = TfidfVectorizer()

    model = make_pipeline(CleanTextTransformer(), vectorizer, model)
    model.fit(text, labels)
    return model, classes

def test_model(model, data, classes):
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)

    text, labels = data

    predictions = model.predict(text)

    print('Accuracy:', accuracy_score(labels, predictions, normalize=True))
    print(classification_report(labels, predictions))

    mat = confusion_matrix(labels, predictions)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig('confusion.pdf')
    plt.show()


def predict_category(s, model):
    pred = model.predict([s])
    return pred[0]