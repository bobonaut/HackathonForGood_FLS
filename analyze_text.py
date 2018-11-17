#!/usr/bin/env python3

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

filename = './models/propaganda_text_model.sav'

nltk.download('stopwords'),nltk.download('porter_test')
stop_words = nltk.corpus.stopwords.words('english')

model = joblib.load(filename)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
ps = nltk.PorterStemmer()


def normalize_message(txt):
    z = re.sub("[^a-zA-Z]", " ", str(txt))
    z = re.sub(r'[^\w\d\s]', ' ', z)
    z = re.sub(r'\s+', ' ', z)
    z = re.sub(r'^\s+|\s+?$', '', z.lower())
    return ' '.join(ps.stem(term)
                    for term in z.split()
                    if term not in set(stop_words)
                    )


def predict(message):
    normalized_message = normalize_message(message)

    if model.predict(vectorizer.transform([normalized_message])):
        return 1
    else:
        return 0


print('Trump: ' + predict('Trump'))
print('Trump & Putin: ' + predict('Trump & Putin'))
print('Nato: ' + predict('Nato'))
print('Sergey: ' + predict('Sergey'))
print('Terrorist: ' + predict('Terrorist'))
print('Stop NATO: ' + predict('Stop NATO'))
print('Stop war: ' + predict('Stop war'))
