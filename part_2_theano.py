import csv
import nltk
import itertools
import numpy as np
import datetime
import sys

from rnn_theano import RNNTheano
from utils import load_model_parameters_theano, save_model_parameters_theano

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

model = RNNTheano(vocabulary_size, hidden_dim=50)
load_model_parameters_theano('./trained-model-theano.npz', model)

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
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
