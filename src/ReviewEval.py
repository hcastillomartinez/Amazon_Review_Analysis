import glob
import os

import numpy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import pandas as pd

def toClass(x):
    if x > 15:
        return 5
    else:
        return int(x/5)

def toTier(x):
    if x <= 5:
        return 0
    if x <= 10:
        return 1
    if x <= 15:
        return 2
    if x <= 20:
        return 3
    if x > 20:
        return 4
    return 7


maxlen = 200 # 150
embedding_dim = 100 # 150
maxWords = 2000 # 4000
testSize = 0.5
epochs = 2 # 2
batchSize = 10

'''
model = Sequential()
model.add(layers.Embedding(input_dim=maxWords, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))#
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(20, activation='relu'))15
model.add(layers.Dense(10, activation='relu'))10
# Output
model.add(layers.Dense(5, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model_json = model.to_json()
with open("assets/ReviewEval_Model.json", "w") as json_file:
    json_file.write(model_json)
'''
reviewFiles = []
for file in glob.glob("assets/*.tsv"):
    fileFrame = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    print("loaded", len(fileFrame), "reviews from", file)
    reviewData = fileFrame[["review_body", "helpful_votes"]]

    # Data cleaning
    # Remove all reviews with null helpful votes or review text
    reviewData.dropna()
    # Remove reviews without an actual review body (just a rating)
    # reviewData = reviewData[reviewData.line_race != ""]

    sum = 0
    for i in reviewData["helpful_votes"]:
        sum += i
    mean = sum / len(reviewData["helpful_votes"])
    print("Avg", mean)

    hist = {}
    stdSum = 0
    for i in reviewData["helpful_votes"]:
        hist[toTier(i)] = hist.get(toTier(i), 0) + 1
        stdSum += (mean - toTier(i)) ** 2
    stddev = numpy.math.sqrt(stdSum * (1 / len(reviewData["helpful_votes"])))
    print("Stddev", stddev)
    print(hist)

    #reviewData["review_body"] = reviewData["review_body"].map(str)
    # Assign a tier to the review based on the number of helpful votes
    reviewClasses = reviewData["helpful_votes"].map(toTier)
    # Fetch all review bodies
    reviews = reviewData['review_body'].map(str).values
    # Convert review tiers to a categorical format for training and evaluation
    scores = to_categorical(reviewClasses)

    reviewTraining, reviewTest, helpTraining, helpTest = train_test_split(reviews, scores, test_size=testSize, random_state=1000)
    print("training reviews:", len(reviewTraining))

    # Embedding
    tokenizer = Tokenizer(num_words=maxWords)
    tokenizer.fit_on_texts(reviewTraining)

    # Find embedding values for reviews
    X_train = tokenizer.texts_to_sequences(reviewTraining)
    X_test = tokenizer.texts_to_sequences(reviewTest)

    # Normalize review body by padding for truncating
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    print("processed reviews:", X_train.shape[0])
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab size:", vocab_size)

    #Create model
    model = Sequential()
    model.add(layers.Embedding(input_dim=maxWords, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(128, 5, activation='relu'))  #
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    # Output
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # categorical_crossentropy binary_crossentropy
    print(model.summary())

#    reviewTraining, reviewTest, helpTraining, helpTest = train_test_split(reviews, scores, test_size=testSize, random_state=1000)

    model.fit(X_train, helpTraining, epochs=epochs, verbose=False, validation_data=(X_test, helpTest), batch_size=batchSize)
    #model.fit(X_test, helpTest, epochs=1, verbose=False, validation_data=(X_train, helpTraining), batch_size=batchSize)
    #model.fit(X_train, helpTraining, epochs=1, verbose=False, validation_data=(X_test, helpTest), batch_size=batchSize)
    #model.fit(X_test, helpTest, epochs=1, verbose=False, validation_data=(X_train, helpTraining), batch_size=batchSize)


    # Testing
    loss, accuracy = model.evaluate(X_train, helpTraining, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, helpTest, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    '''
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
    fileName = os.path.basename(file)
    # serialize weights to HDF5
    model.save_weights("assets/weights/model_for_"+fileName+".h5")
    print("Saved model to disk")

#reviewData = pd.concat(reviewFiles)
'''
#"assets/modified_amazon_reviews_us_Major_Appliances_v1_00.tsv" "assets/modified_amazon_reviews_us_Gift_Card_v1_00.tsv"
file = "assets/modified_amazon_reviews_us_Lawn_and_Garden_v1_00.tsv"
fileFrame = pd.read_csv(file, delimiter='\t', encoding='utf-8')
reviewData = fileFrame[["review_body", "helpful_votes", "total_votes"]]

print("total reviews:", len(reviewData))

# Cleaning Data
previousTotal = len(reviewData)

# Drop all null columns
print("Removing null rows")
reviewData.dropna()
print("dropped", previousTotal-len(reviewData), "reviews")


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
scores = to_categorical(reviewClasses)
#print(scores)

reviewTraining, reviewTest, helpTraining, helpTest = train_test_split(reviews, scores, test_size=0.5, random_state=1000)
print("training reviews:", len(reviewTraining))

# Embedding

maxlen = 200
embedding_dim = 100
maxWords = 2000

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
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
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

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_for_"+file+".h5")
print("Saved model to disk")

'''
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