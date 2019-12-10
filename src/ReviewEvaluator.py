import pickle
import sys

from keras.engine.saving import model_from_json
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from pandas._libs import json
from keras.preprocessing.sequence import pad_sequences
import pickle

#argText = "I hate this item" #"#reviewData.loc[[16]]

argText = sys.argv[1]
weights = sys.argv[2]
tokenizer = sys.argv[3]

print("Evaluating:", argText)
print("With weights from:", weights)
print("With tokenizer from:", tokenizer)

# Pre process input
#tokenizer = Tokenizer(num_words=200)
#tokenizer.fit_on_texts(argText)

'''
tokenizerFile = open(tokenizer, 'r')
data = tokenizerFile.read()
tokenizer = tokenizer_from_json(data)
'''
with open(tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizedText = tokenizer.texts_to_sequences(argText)
tokenizedText = pad_sequences(tokenizedText, padding='post', maxlen=200)

print(tokenizedText)

# Load model
modelFile = open("review_eval_model.json", 'r')
modelJSON = modelFile.read()
modelFile.close()
model = model_from_json(modelJSON)
model.load_weights(weights)

val = model.predict(tokenizedText)
print(val)
