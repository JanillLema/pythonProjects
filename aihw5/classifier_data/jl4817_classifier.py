"""
WITHOUT THE STOP WORD FILE
m=k=1
training set: 0.9917040358744394
test set: 0.9820466786355476
held-out: 0.9784560143626571


WITH THE STOP WORD FILE
m=k=1
training set: 0.9910313901345291
test set: 0.9820466786355476
held-out: 0.9748653500897666

PART 4:
m= 2
k= .4
training set: 0.9905829596412556
test set: 0.9802513464991023
held-out: 0.9748653500897666
"""

import sys
import string
import math
import operator

class NbClassifier(object):

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):
        stop_words = set()
        input = open(stopword_file, 'r')
        line = input.readline()
        while line:
            word = self.extract_words(line)

            stop_words.add(word[0])
            line = input.readline()

        self.attribute_types = self.attribute_types.difference(stop_words)

    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=2):
        self.attribute_types = set()
        word_count = {}
        # read the file
        input = open(training_filename, 'r')
        i = input.readline()
        while i:
            # split the line based on tab
            line = i.split("\t")
            line = line[1]
            words_list = self.extract_words(line)
            # place word in a dictionary that has the number of times it appears as its value
            for word in words_list:
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] = word_count[word] + 1
            i = input.readline()

        # for each item in the dictionary, if the value is >= than m, add the key to the set
        for k,v in word_count.items():
            if v >= m:
                self.attribute_types.add(k)




    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, z=.04):
        self.label_prior = {}
        self.word_given_label = {}

        total_y = {}

        # populates label_prior
        input = open(training_filename, 'r')
        i = input.readline()
        while i:
            # split the line based on tab
            line = i.split("\t")
            # populates the keys
            if line[0] not in self.label_prior:
                self.label_prior[line[0]] = 0
            # used to find total(y)
            if line[0] not in total_y:
                total_y[line[0]] = 0

            words_list = self.extract_words(line[1])
            # populates values as the number of words under each label
            for word in words_list:
                self.label_prior[line[0]] = self.label_prior[line[0]] + 1
                total_y[line[0]] = total_y[line[0]] + 1
                if word in self.attribute_types:
                    if (word,line[0]) not in self.word_given_label:
                        self.word_given_label[(word,line[0])] = 1
                    else:
                        self.word_given_label[(word, line[0])] = self.word_given_label[(word, line[0])] + 1

            i = input.readline()

        # adds in features to the dict that do not appear in other labels
        for w in self.attribute_types:
            for u in total_y.keys():
                if (w,u) not in self.word_given_label.keys():
                    self.word_given_label[(w,u)] = 0


        total = 0
        for k,v in self.label_prior.items():
            total = total + v
        # final value calculation
        for k,v in self.label_prior.items():
            self.label_prior[k] = self.label_prior[k] / total
        for elem in self.word_given_label:
            total_y_val = 0
            for a,b in total_y.items():
                if elem[1] == a:
                    total_y_val = b+1

            self.word_given_label[elem] = self.word_given_label[elem] + z
            self.word_given_label[elem] = self.word_given_label[elem] / ((len(self.attribute_types)*z) + total_y_val)
        print(self.label_prior)


    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):


        probabilities = {}
        line = self.extract_words(text)

        for elem in self.label_prior.keys():
            probabilities[elem] = math.log(self.label_prior[elem])

        for word in line:
            if word in self.attribute_types:
                for label in probabilities.keys():
                    probabilities[label] = probabilities[label] + math.log(self.word_given_label[(word, label)])

        return probabilities


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):
        total_lines = 0
        correct = 0
        input = open(test_filename, 'r')
        i = input.readline()
        while i:
            total_lines = total_lines + 1
            # split the line based on tab
            line = i.split("\t")

            a = self.predict(line[1])
            k = max(a.items(), key=operator.itemgetter(1))[0]
            if k == line[0]:
                correct = correct+1

            i = input.readline()

        return correct / total_lines


if __name__ == "__main__":

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])

    print(classifier.evaluate("train.txt"))
    print(classifier.evaluate("test.txt"))
    print(classifier.evaluate("dev.txt"))






