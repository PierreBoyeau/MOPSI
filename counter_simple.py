import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math


"""Expectation and variance computed thanks to an approximation based on Ergodic Theorem."""


class Counter_simple:
    """Class used to determine the rarity of a given subsequence in a DNA sequence.
        Model used: Markov chains"""
    def __init__(self, sequence_lenght):
        self.occurrences = None
        self.vocab = None
        self.seq_len = sequence_lenght

    def make_chunks(self, s):
        return [s[i:i + width] for width in {1, 2, self.seq_len - 1, self.seq_len}
                for i in range(len(s) - width + 1)]

    def learn(self, sequence):
        vectorizer = CountVectorizer(tokenizer=self.make_chunks, lowercase=False)
        self.occurrences = vectorizer.fit_transform(sequence)
        self.occurrences = self.occurrences.toarray()
        self.occurrences = self.occurrences[0]
        self.vocab = vectorizer.get_feature_names()
        self.vocab = np.array(self.vocab)

    def occ(self, word):
        return int(self.occurrences[self.vocab == word])

    def expectation(self, word):
        """Estimator computed thanks to the Ergodic Theorem"""
        return self.occ(word[:-1])*self.occ(word[-2:]) / self.occ(word[-2])

    def variance(self, word):
        """Estimation of variance of expected occurences for given word"""
        frac = self.occ(word)
        term1 = 1 - self.occ(word[:-1])/self.occ(word[-2])
        term2 = 1 - self.occ(word[-2:])/self.occ(word[-2])
        return frac * term1 * term2

    def p_score(self, word):
        return (self.occ(word) - self.expectation(word)) / math.sqrt(self.variance(word))


# --- Examples (see Count_m)
