from datetime import datetime
import glob
import os

import numpy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle

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


def toTier2(x):
    if x <= 4:
        return 0
    if x <= 16:
        return 1
    if x > 16:
        return 2
    return 7

def getVotesOverTimeColumn(data):
    votePerAge = []
    now = datetime.now()
    for row in data.iterrows():
        body = row[0]#row['review_body']
        votes = row[1][1]#row['helpful_votes']
        timestamp = row[1][2]#row['review_date']
        age = now - datetime.strptime(timestamp, "%Y-%m-%d")
        votePerAge.append(votes/age.days)
    return votePerAge

def clamp(row):
    body = row['review_body']
    tier = toTier2(row['helpful_votes'])
    tierCounts[tier] = tierCounts[tier] + 1
    if tierCounts[tier] > tierLimits[tier]:
        row['review_body'] = -1#None
        row['helpful_votes'] = -1#None
        return row
    else:
        return row


maxlen = 150 # 150
embedding_dim = 100 # 150
maxWords = 2000 # 4000
testSize = 0.5
epochs = 3 # 2
batchSize = 10

limit = 5000

reviewFiles = []
for file in glob.glob("assets/*.tsv"):
    fileName = os.path.basename(file)
    print("-------working on:", fileName)
    fileFrame = pd.read_csv(file, delimiter='\t', encoding='utf-8')
    print("loaded", len(fileFrame), "reviews from", file)
    reviewData = fileFrame[["review_body", "helpful_votes", "review_date"]]

    print(int(len(fileFrame)*(1-testSize)), "reviews available for training")

    # Data cleaning

    # Remove all reviews with null helpful votes or review text
    reviewData.dropna()

    # Assign a tier to the review based on the number of helpful votes
    reviewClasses = reviewData["helpful_votes"].map(toTier2)

    # Find summary stats and output histogram
    sum = 0
    for i in reviewClasses:
        sum += i+1
    mean = sum / len(reviewClasses)
    print("Avg", mean)

    hist = {}
    stdSum = 0
    for i in newColumn:
        hist[i] = hist.get(i, 0) + 1
        stdSum += (mean - i) ** 2

    stddev = numpy.math.sqrt(stdSum * (1 / len(reviewClasses)))
    print("Stddev", stddev)
    print(hist)

    limit = hist[2]*3
    print("capping max reviews per tier to", limit)
    tierLimits = [limit, limit, limit]
    tierCounts = [0, 0, 0]

    # Remove excess reviews
    reviewData = reviewData.apply(clamp, axis=1)
    reviewData = reviewData[reviewData.helpful_votes != -1]
    print(len(reviewData), "reviews remaining after trimming excess")

    # Assign a tier to the review based on the number of helpful votes
    reviewClasses = reviewData["helpful_votes"].map(toTier2)

    # Fetch all review bodies
    reviews = reviewData['review_body'].map(str).values
    # Convert review tiers to a categorical format for training and evaluation
    scores = to_categorical(reviewClasses)

    # Split data
    reviewTraining, reviewTest, helpTraining, helpTest = train_test_split(reviews, scores, test_size=testSize, random_state=1000)
    print("training reviews:", len(reviewTraining))

    # Embedding
    tokenizer = Tokenizer(num_words=maxWords)
    tokenizer.fit_on_texts(reviewTraining)

    # Save tokenizer
    print("Tokenizer saved as:", "assets/weights/tokenizer_for_"+fileName+".pickle")
    with open("assets/weights/tokenizer_for_"+fileName+".pickle", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Find embedding values for reviews
    X_train = tokenizer.texts_to_sequences(reviewTraining)
    X_test = tokenizer.texts_to_sequences(reviewTest)

    # Normalize review body by padding for truncating
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    print("processed reviews:", X_train.shape[0])
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab size:", vocab_size)

    # Create model
    model = Sequential()
    model.add(layers.Embedding(input_dim=maxWords, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(50))
    # Output
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    #print(model.summary())
    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())

    model.fit(X_train, helpTraining, epochs=epochs, verbose=2, validation_data=(X_test, helpTest), batch_size=batchSize)

    # Testing
    loss, accuracy = model.evaluate(X_train, helpTraining, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, helpTest, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    #tier 0 1 helpful votes
    testReview = "great service and product was just as i wanted"
    print(testReview)
    testReviewConv = tokenizer.texts_to_sequences([testReview])
    testReviewConv = pad_sequences(testReviewConv, padding='post', maxlen=maxlen)
    #print(testReviewConv)
    val = model.predict(testReviewConv)
    print(val)

    #tier 1 8 helpful votes
    testReview = "This piece of crap stopped working a few days ago. It simply will not cool below 70 degrees Fahrenheit! I have tried moving it to a different location and also I unplugged for a few hours hoping for a &#34;reset.&#34; Nothing works! Don't buy it. I'm shocked that Amazon continues to sell such low quality items."#8 tier 1
    print(testReview)
    testReviewConv = tokenizer.texts_to_sequences([testReview])
    testReviewConv = pad_sequences(testReviewConv, padding='post', maxlen=maxlen)
    #print(testReviewConv)
    val = model.predict(testReviewConv)
    print(val)

    #tier 2 36 helpful votes
    testReview = "I have had my Danby washer for a few months now and I love it.  It supplements my mini washer and dryer unit in the apartment.  Whereas it used to take all day to do laundry, now I can get 3 loads done at once.  I can wash 2 loads in the Danby in the time it takes one load to wash and rinse in the small washing machine that comes with the apartment.  I put anything that does not need to be dried in the dryer on laundry racks.  Things that will need to be ironed anyway and socks, underwear, pajamas and such items go on the rack.  The spin side of the washer extracts so much water that it only takes a few hours to air dry.<br />This unit takes some labor as wet clothing needs to be transferred to the spin side and then rinsed and transferred to spin again.  A person needs to know that it is not an automatic machine that you can walk away from (except when it is washing and then the timer is set).  The fill hose on top does not fit the faucet on my kitchen sink but it doesn't take much to fill the top with an alternative hose or bucket if you have to.  I drain the water into the sink, making sure that I am there to hold the hose or affix it to something so that it doesn't slip and drain water all over the floor.  If you don't mind doing some of the work, it is a nice little machine.  I have washed sheets, towels,  and a few pairs of jeans in it with no problems.  If you live in an apartment or have no hookups for a regular machine, I would recommend this.  When I am finished with it, I just roll it away from the sink and store it against the wall.  If there is any water left in the drain hose, it can be drained into a bucket before being hooked on the side of the machine."
    print(testReview)
    testReviewConv = tokenizer.texts_to_sequences([testReview])
    testReviewConv = pad_sequences(testReviewConv, padding='post', maxlen=maxlen)
    #print(testReviewConv)
    val = model.predict(testReviewConv)
    print(val)

    # serialize weights to HDF5
    model.save_weights("assets/weights/model_for_"+fileName+".h5")
    print("Saved model to disk")
