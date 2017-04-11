import theano
import csv
import nltk
import itertools
import numpy as np
from rnn_numpy import RNNNumpy

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("Reading CSV file....")

with open('reddit-comments-2015-08.csv', 'r', encoding="utf-8") as reddit_file:
    reader = csv.reader(reddit_file, skipinitialspace=True)
    reader.__next__() # To skip the first line?

    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print("Parsed %d sentences." % (len(sentences)))

tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
 
# Create the training data

# Note that the length of each sentence is different
# X_train - every words of each sentence except for the last one
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
# y_train - every words except for the first one
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

model = RNNNumpy(vocabulary_size)
o, s = model.forward_propagation(X_train[10]) # 10th training example

print(o.shape)
print(o)

predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)

print("Expected loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)

model.numpy_sgd_step(X_train[10], y_train[10], 0.005)
