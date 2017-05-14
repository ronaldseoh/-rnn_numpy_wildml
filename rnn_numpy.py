import numpy as np
import operator
from utils import softmax

class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

        np.random.seed(10)

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # U is the weight matrix for words
        # Each column will be picked for hidden state calculation
        self.U = np.random.uniform(
                        - np.sqrt(1./word_dim),
                        np.sqrt(1./word_dim),
                        (hidden_dim, word_dim)
                    )

       # V is the weight matrix for outputs
       # V will be multiplied to the hidden state s_t
        self.V = np.random.uniform(
                        - np.sqrt(1./hidden_dim),
                        np.sqrt(1./hidden_dim),
                        (word_dim, hidden_dim)
                    )

        # W is the weight matrix for the PREVIOUS hidden state s_(t-1)
        self.W = np.random.uniform(
                        - np.sqrt(1./hidden_dim),
                        np.sqrt(1./hidden_dim),
                        (hidden_dim, hidden_dim)
                    )

    def forward_propagation(self, x):

        # This would be (the number of words in the sentence 'x' - 1),
        # where every one of x provides output
        T = len(x)

        # hidden states
        # this isn't the one that gets trained; just calculated
        # based on the trained U and W
        s = np.zeros((T+1, self.hidden_dim)) # +1 for the last state

        # the last state initialized with zeros
        # to calculate the first hidden state
        s[-1] = np.zeros(self.hidden_dim)

        # (# of words in a single sentence, 8000)
        o = np.zeros((T, self.word_dim))

        # For every word in this sentence
        for t in np.arange(T):
            # state for each word
            s[t] = np.tanh(
                # The column of U that covers the word number x[t]
                # (applies to all the training sentences(examples))
                self.U[:, x[t]] # (hidden_dim, 1)

                # Weight given to previous state
                + self.W.dot(s[t-1]) # (hidden_dim, 1)
            )

            # predict the next word on t+1 based on s[t]
            o[t] = softmax(self.V.dot(s[t])) #(word_dim, 1)

        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)

        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):

        L = 0

        for t in np.arange(len(y)):

            o, s = self.forward_propagation(x[t])

            # Note that o was created with the dimension of (T, self.word_dim)
            # think of this as diagonally going through output!!
            correct_word_predictions = o[
                # we can't use ':' here because we are now cherry-picking
                # the probability assigned to correct answers for each word slot
                # We use integer array indexing, not slicing,
                # to construct a custom array, instead of having a subarray
                # of 'o'.
                # Check out http://cs231n.github.io/python-numpy-tutorial/
                # np.arange() : Return evenly spaced values within a given
                # interval.

                # From 0 to (len(y[i]) - 1)
                # # of words in X = # of words in Y = # of outputs
                np.arange(len(x[t]))
                ,y[t] # this isn't scalar; one sentence
            ]

            # -1 to make the value of loss positive number
            L += -1 * np.sum(
                # if the probability assigned to the correct words is low,
                # this will be way down the negative territory
                np.log(correct_word_predictions)
            )

        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))

        # Loss averaged by total number of words in y (not the whole vocab set!)
        return self.calculate_total_loss(x, y) / N

    # Back-Propagation Through Time = BPTT
    def bptt(self, x, y):
        T = len(y)

        o, s = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.

        for t in np.arange(T)[::1]: # get items by index incremented by 1
            dLdV += np.outer(delta_o[t], s[t].T)

            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t

                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)

        parameter_names = ['U', 'V', 'W']

        # enumerate provides serial numbers to each object out of iterables
        for index, param_name in enumerate(parameter_names):
            # Gets the attribute 'pname' from self,
            # which would be a RNNNumpy instance
            current_parameter = operator.attrgetter(param_name)(self)

            print("Performing gradient check for parameter %s with size %d."
                    % (param_name, np.prod(current_parameter.shape))
            )

            # np.nditer(): Efficient multi-dimensional iterator object
            # to iterate over arrays.
            it = np.nditer(
                    current_parameter,
                    # Flags to control the behavior of the iterator
                    flags=['multi_index'], # a tuple of indices
                    # This is a list of flags for each operand.
                    # At minimum, one of “readonly”, “readwrite”,
                    # or “writeonly” must be specified.
                    op_flags=['readwrite']
                )

            while not it.finished:
                ix = it.multi_index

                # Temporaily change the parameters to calculate
                # the estimated gradient
                original_value = current_parameter[ix]

                current_parameter[ix] = original_value + h
                loss_plus = self.calculate_total_loss([x], [y])

                current_parameter[ix] = original_value - h
                loss_minus = self.calculate_total_loss([x], [y])

                estimated_gradient = (loss_plus - loss_minus) / (2 * h)

                # Back to original value
                current_parameter[ix] = original_value

                # Get the actual gradient values from bptt
                backprop_gradient = bptt_gradients[index][ix]

                relative_error = np.abs(backprop_gradient-estimated_gradient) \
                / (np.abs(backprop_gradient) + np.abs(estimated_gradient))

                # If the error between estimated_gradient and backprop_gradient
                # is too large, print the details
                if relative_error > error_threshold:
                    print("GRADIENT CHECK ERROR: parameter=%s ix=%s" % (param_name, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return

                it.iternext()

            print("Gradient check for parameter %s passed." % (param_name))

    def numpy_sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)

        # Learning rate is adjustable between the steps!
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
