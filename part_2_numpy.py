import theano
import csv
import nltk
import itertools
import numpy as np
import datetime
import sys
from rnn_numpy import RNNNumpy

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("Reading CSV file....")

with open('reddit-comments-2015-08.csv', 'r', encoding="utf-8") as reddit_file:
    reader = csv.reader(reddit_file, skipinitialspace=True)
    reader.__next__() # To skip the first line?

    # Split full comments into SENTENCES
    # x is a list with length 1 - need x[0] to access the full string
    # Then nltk.sent_tokenize() makes each string into a list of SENTENCES
    # itertools.chain combines separate iterators into a single iterator
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    # After this, 'sentences' will be a list full of sentences with tokens attached
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print("Parsed %d sentences." % (len(sentences)))

# Then nltk.word_tokenize() makes each string with a sentence in 'sentences'
# into a list of WORDS
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Mash up all the word list from all the sentences and calculate
# the frequency of every single words using nltk.FreqDist()
# len(word_freq) == 65752
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# We only deal with the top 7999 frequently used words, and treat
# everything else as a same word "UNKNOWN_TOKEN"
# word_freq.most_common() returns the list that contains
# n most common elements and their counts,
# ordered by descending frequencies.
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab] # This is just a list of words, created to give serial numbers to words
index_to_word.append(unknown_token) # The last word in index_to_word is "UNKNOWN_TOKEN"
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with "UNKNOWN_TOKEN"
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
# Note that the length of each sentence is different
# X_train - every words of each sentence except for the last one
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
# y_train - every words except for the first one
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

############################
# Test FORWARD PROPAGATION #
############################
model_test_1 = RNNNumpy(vocabulary_size)
o, s = model_test_1.forward_propagation(X_train[10]) # 10th training example
print(o.shape) # (45, 8000)
print(o)

# Calculate a prediction by forward-propagating with the current weight values
# even though they wouldn't be optimal
predictions = model_test_1.predict(X_train[10])
print(predictions.shape)
print(predictions)

print("Expected loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model_test_1.calculate_loss(X_train[:1000], y_train[:1000]))

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)

model.numpy_sgd_step(X_train[10], y_train[10], 0.005)

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0

    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%s')

            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))

            if (len(losses) > 1 and losses[-1][1] > [-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)

            sys.stdout.flush()

        for i in range(len(y_train)):
            model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
