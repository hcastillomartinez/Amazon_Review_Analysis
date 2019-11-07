import glob

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import pandas as pd
'''
reviewFiles = []
for file in glob.glob("assets/*.tsv"):
    fileFrame = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    print("loaded", len(fileFrame), "reviews from", file)
    data = fileFrame[["review_body", "helpful_votes", "total_votes"]]
    data = data[data["review_body"].notnull()]
    data = data[data["helpful_votes"].notnull()]
    data = data[data["total_votes"].notnull()]
    reviewFiles.append(data)

reviewData = pd.concat(reviewFiles)
'''
fileFrame = pd.read_csv("assets/modified_amazon_reviews_us_Major_Appliances_v1_00.tsv", delimiter='\t', encoding='utf-8')
reviewData = fileFrame[["review_body", "helpful_votes", "total_votes"]]
#reviewData = reviewData[reviewData["review_body"].notnull()]
#reviewData = reviewData[reviewData["helpful_votes"].notnull()]
#reviewData = reviewData[reviewData["total_votes"].notnull()]

print("total reviews:", )

# Cleaning Data
previousTotal = len(reviewData)

# Drop all null columns
print("Removing null rows")
reviewData.dropna()
print("dropped", previousTotal-len(reviewData), "reviews")

'''
helpful_Votes = reviewData["helpful_votes"].values

sum = 0
for i in helpful_Votes:
    sum += i
mean = sum/len(helpful_Votes)
print("Avg", mean)

hist = {}
stdSum = 0
for i in helpful_Votes:
    if (i > 15):
        hist[5] = hist.get(5, 0) + 1
    else:
        hist[int(i/5)] = hist.get(int(i/5), 0) + 1
    stdSum += (mean - i)**2
stddev = numpy.math.sqrt(stdSum * (1 / len(helpful_Votes)))
print("Stddev", stddev)
print(hist)
'''

def toClass(x):
    if (x > 15):
        return 5
    else:
        return int(x/5)


reviewData["review_body"] = reviewData["review_body"].map(str)
reviewClasses = reviewData["helpful_votes"].map(toClass)

reviews = reviewData['review_body'].values

# Convert helpful_votes to help tier


def helpTierToNNOutput(x):
    out = [0] * 6
    out[x] = 1
    return out


scores = to_categorical(reviewClasses)
#print(scores)

reviewTraining, reviewTest, helpTraining, helpTest = train_test_split(reviews, scores, test_size=0.5, random_state=1000)
print("training reviews:", len(reviewTraining))

# Embedding

maxlen = 100
embedding_dim = 100
maxWords = 2500

tokenizer = Tokenizer(num_words=maxWords)
tokenizer.fit_on_texts(reviewTraining)

X_train = tokenizer.texts_to_sequences(reviewTraining)
X_test = tokenizer.texts_to_sequences(reviewTest)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print("processed reviews:", X_train.shape[0])
vocab_size = len(tokenizer.word_index) + 1
print("vocab size:", vocab_size)

model = Sequential()
model.add(layers.Embedding(input_dim=maxWords, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))#
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(60, activation='relu'))
# Output
model.add(layers.Dense(6, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, helpTraining, epochs=5, verbose=False, validation_data=(X_test, helpTest), batch_size=10)

# Testing
loss, accuracy = model.evaluate(X_train, helpTraining, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, helpTest, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

testReview = reviewData.loc[[16]]
print(testReview)
review = testReview["review_body"]
print(testReview["review_body"])
testReviewConv = tokenizer.texts_to_sequences(review)
testReviewConv = pad_sequences(testReviewConv, padding='post', maxlen=maxlen)
#print(testReviewConv)
val = model.predict(testReviewConv)
print(val)

'''
vectorizer = CountVectorizer()
vectorizer.fit(reviewTraining)

reviewTraining = vectorizer.transform(reviewTraining)
reviewTest = vectorizer.transform(reviewTest)

classifier = LogisticRegression()
classifier.fit(reviewTraining, helpTraining)
score = classifier.score(reviewTest, helpTest)
print(score)
'''