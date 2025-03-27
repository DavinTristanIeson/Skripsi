test_str = ["The quick brown fox jumped over the lazy dog"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

vectorizer.fit(test_str)

vectorizer2 = CountVectorizer(vocabulary=[*vectorizer.vocabulary_, "hello", "world"])

vectorizer2.fit(test_str)

print(vectorizer2.vocabulary_)
