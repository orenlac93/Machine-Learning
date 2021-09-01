# using python 3.8

import random
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# label of review

class ReviewLabel:
    NEGATIVE = "NEGATIVE REVIEW"
    POSITIVE = "POSITIVE REVIEW"


# review class

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.label = self.get_label()

    def get_label(self):
        if self.score <= 2:
            return ReviewLabel.NEGATIVE
        elif self.score == 3:   # neutral or unclear review (choose to ignore)
            return
        else:
            return ReviewLabel.POSITIVE   # 4 or 5


# collection of reviews to balance between the amount of positive and negative reviews

class ReviewCollection:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_label(self):
        return [x.label for x in self.reviews]

    def balance_positive_and_negative(self):
        negative = list(filter(lambda x: x.label == ReviewLabel.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.label == ReviewLabel.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


# loading the data set

# file_name = './data/reviews_size_1000.json'
# file_name = './data/reviews_size_7000.json'
file_name = './data/reviews_size_10000.json'


reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))


# preparing the train and test data

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewCollection(training)

test_container = ReviewCollection(test)

train_container.balance_positive_and_negative()
train_x = train_container.get_text()
train_y = train_container.get_label()

test_container.balance_positive_and_negative()
test_x = test_container.get_text()
test_y = test_container.get_label()

print('balanced data: ')

print('positives: ', train_y.count(ReviewLabel.POSITIVE))
print('negatives: ', train_y.count(ReviewLabel.NEGATIVE))

print('')


# use a kind of 'bag of words' implemented as vectors

tfidf_vectorizer = TfidfVectorizer()   # Tfidf vevtorizer

train_x_vectors = tfidf_vectorizer.fit_transform(train_x)
test_x_vectors = tfidf_vectorizer.transform(test_x)


count_vectorizer = CountVectorizer()   # Count vevtorizer

count_train_x_vectors = count_vectorizer.fit_transform(train_x)
count_test_x_vectors = count_vectorizer.transform(test_x)


print('data as review text: ', train_x[0])
print('data as vector: ', train_x_vectors[0].toarray())
print('')


# simple test with each algorithm
print('random review from the test to check the prediction: ')
print(test_x[0])
print('')

print('each algorithm prediction: ')
print('')

# SVM

clf_svm = svm.SVC(kernel='linear')   # based on tfidf vectorizer
count_clf_svm = svm.SVC(kernel='linear')   # based on count vectorizer

clf_svm.fit(train_x_vectors, train_y)
count_clf_svm.fit(count_train_x_vectors, train_y)

print('SVM')
print(clf_svm.predict(test_x_vectors[0]))
print('')


# decision tree

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

print('decision tree')
print(clf_dec.predict(test_x_vectors[0]))
print('')


# naive bayes

clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)

print('naive bayes')
print(clf_gnb.predict(test_x_vectors[0]))
print('')


# logistic regression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

print('logistic regression')
print(clf_log.predict(test_x_vectors[0]))
print('')


# each method Accuracy

print('check the accuracy for all the test data: ')
print('')

print('SVM Accuracy: ', clf_svm.score(test_x_vectors, test_y))
print('decision tree Accuracy: ', clf_dec.score(test_x_vectors, test_y))
print('naive bayes Accuracy: ', clf_gnb.score(test_x_vectors, test_y))
print('logistic regression Accuracy: ', clf_log.score(test_x_vectors, test_y))
print('')


# check accuracy on the positive alone and negative alone

print('positive reviews accuracy and negative reviews accuracy: ')
print('positive   |   negative')
print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[ReviewLabel.POSITIVE, ReviewLabel.NEGATIVE]))
print('')


# test prediction on specific data
print('simple short text test: ')
test_set = ['it was a great book', "bad book do not buy", 'it was a waste of time']
print(test_set)
new_test = tfidf_vectorizer.transform(test_set)

print(clf_svm.predict(new_test))

# test accuracy differences between two vectorizer methods

print('')
print('compare between to vectorizer methods:')
print('Tfidf Vectorizer Accuracy: ', clf_svm.score(test_x_vectors, test_y))
print('Count Vectorizer Accuracy: ', count_clf_svm.score(count_test_x_vectors, test_y))
