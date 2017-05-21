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
# the frequency of every single word using nltk.FreqDist()
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

print()

print("############################")
print("# Test FORWARD PROPAGATION #")
print("############################")
model_test_forward_prop = RNNNumpy(vocabulary_size)

# Using 10th training example
o, s = model_test_forward_prop.forward_propagation(X_train[10])

print(o.shape) # (45, 8000)
print(o)

# Try calculating a prediction by forward-propagating 
# with the current weight values
# even though they obviously would be very far from optimal
predictions = model_test_forward_prop.predict(X_train[10])
print(predictions.shape)
print(predictions)

# According to the tutorial: Since we have (vocabulary_size) words, so each word
# should be predicted, on average, with probability 1/C, which would yield
# a loss of L = -1/N * N * log(1/C) = log(C)
print("Expected loss for random predictions: %f" % np.log(vocabulary_size))
print("Actual loss: %f" % model_test_forward.calculate_loss(X_train[:1000], y_train[:1000]))

print()

print("#######################")
print("# Test GRADIENT CHECK #")
print("#######################")
# To avoid performing millions of expensive calculations we use
# a smaller vocabulary size for checking.
grad_check_vocab_size = 100

np.random.seed(10) # re-seed the generator
model_test_grad_check = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model_test_grad_check.gradient_check([0,1,2,3], [1,2,3,4])

print("##########################")
print("# Test a single SGD STEP #")
print("##########################")
np.random.seed(10)
model_test_sgd_step = RNNNumpy(vocabulary_size)
model_test_sgd_step.sgd_step(X_train[10], y_train[10], 0.005)

# Train on a small subset of the data to see what happens
print("####################################")
print("# Test TRAINING on a small dataset #")
print("####################################")
np.random.seed(10)

model_training_small = RNNNumpy(vocabulary_size)

# Stochastic Gradient Descent Algorithm
def train_with_sgd(
    model, X_train, y_train, learning_rate=0.005, nepoch=100,
    evaluate_loss_after=5
):
    losses = []
    num_examples_seen = 0

    for epoch in range(nepoch):
        # After 'evalulate_loss_after' number of steps,
        # check the amount of loss and adjust the learning rate if needed
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%s')

            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" 
                    % (time, num_examples_seen, epoch, loss)
            )

            # If the loss just got bigger in the last epoch,
            # decrease the learning rate
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)

            sys.stdout.flush()

        # Train the model with the training data
        # This loop goes through all the examples in the given training set.
        # So we are basically performing 'nepoch' batches of 
        # sgd steps over the whole training set, 
        # each step with a single example.
        for i in range(len(y_train)):
            # Reminder: X_train[i] is a single sentence
            # We are performing the stochastic gradient descent algorithm!
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

    return losses

# Perform the training
losses_training_small = train_with_sgd(
            model_training_small,
            X_train[:100], y_train[:100],
            nepoch=10, evaluate_loss_after=1
        )

print("#####################################")
print("# Test TRAINING on a bigger dataset #")
print("#####################################")
np.random.seed(10)

model = RNNNumpy(vocabulary_size)

losses = train_with_sgd(
            model,
            X_train[:1000], y_train[:1000],
            nepoch=10, evaluate_loss_after=1
        )

print("##############################")
print("# Test a SENTENCE GENERATION #")
print("##############################")

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]

    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs, hidden_state = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]

        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(n=1, pvals=next_word_probs[0])
            sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)

    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)

    print(" ".join(sent))
