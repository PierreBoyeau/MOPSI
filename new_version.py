import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


class Counter_m:
    """Class used to determine the rarity of a given subsequence in a DNA sequence.
    Model used: Markov chains of order m. Supposedly more precise that Counter_simple, but also
    longer to compute"""
    def __init__(self, sequence_lenght):
        self.seq_len = sequence_lenght
        self.m_len = self.seq_len - 2
        self.occurrences = None
        self.vocab = None
        self.alphabet = ['A', 'T', 'C', 'G']

    def make_chunks_word(self, s):
        return [s[i:i + width] for width in {self.m_len, self.m_len + 1}
                for i in range(len(s) - width + 1)]

    def make_chunks(self, s):
        return [s[i:i + width] for width in {self.m_len, self.m_len + 1, self.seq_len}
                for i in range(len(s) - width + 1)]

    def learn(self, sequence):
        """Fit the model to the learning set. Must be a LOCK CAPS string, such as 'ATCGATCGATCGATCT' """
        vectorizer = CountVectorizer(tokenizer=self.make_chunks, lowercase=False)
        self.occurrences = vectorizer.fit_transform(sequence)
        self.occurrences = self.occurrences.toarray()
        self.occurrences = self.occurrences[0]
        self.vocab = vectorizer.get_feature_names()
        self.vocab = np.array(self.vocab)

    def expectation_gaussian(self, word):
        """Computes estimated expectancy for any word such as len(word) >= m_len - 2
        Expects 3.X Python versions"""
        word_len = len(word)
        numerator = 1
        for j in range(0, word_len-self.m_len):
            word1 = word[j:j+self.m_len+1]
            numerator *= int(self.occurrences[self.vocab == word1])
        denom = 1
        for j in range(1, word_len-self.m_len):
            word1 = word[j:j+self.m_len]
            denom *= int(self.occurrences[self.vocab == word1])
        return numerator/denom  # Expect Python 3.X versions

    def __variance_gaussian(self, word):
        """This function is now useless. Please ignore"""
        expect = self.expectation_gaussian(word)
        variance = expect

        # Computation of the independant term
        indep_cst = 0
        for d in range(0, self.seq_len-self.m_len-1):
            if word[0:d+1] == word[self.seq_len-1-d:]:
                word_repeted = ''.join(word[:d+1]+word[:self.seq_len])  # Pas de probleme avec la longueur, toujours bonne
                indep_cst += self.expectation_gaussian(word_repeted)
        variance += (2*indep_cst)

        # Computation of the quadratic term
        word_vector = CountVectorizer(tokenizer=self.make_chunks_word, lowercase=False)
        word_occ = word_vector.fit_transform([word])
        word_occ = word_occ.toarray()
        word_occ = word_occ[0]
        word_vocab = word_vector.get_feature_names()
        word_vocab = np.array(word_vocab)

        quad_cst = 0
        for subword in word_vocab:
            if len(subword) == self.m_len:
                apparitions = [int(word_occ[word_vocab == subword+k]) for k in self.alphabet if (subword+k in word_vocab)]
                num = sum(apparitions)
                quad_cst += (num**2)/int(self.occurrences[self.vocab == subword])
            elif len(subword) == self.m_len+1:
                quad_cst -= (int(word_occ[word_vocab == subword])**2)/\
                            int(self.occurrences[self.vocab == subword])

        apparitions = [int(word_occ[word_vocab == word[0:self.m_len]+k])
                       for k in self.alphabet if (word[0:self.m_len]+k in word_vocab)]
        num = sum(apparitions)
        quad_cst += (1 - 2*num)/int(self.occurrences[self.vocab == word])
        variance += (expect*expect)*quad_cst

        return variance

    def variance_gaussian_max(self, word):
        """Computes variance of word accordingly to our model"""
        expect = self.expectation_gaussian(word)
        occ_word1 = int(self.occurrences[self.vocab == word[1:self.seq_len - 1]])
        occ_word_end = int(self.occurrences[self.vocab == word[1:self.seq_len]])
        occ_word_beg = int(self.occurrences[self.vocab == word[0:self.seq_len - 1]])
        result = expect/(occ_word1**2)*(occ_word1-occ_word_end)*(occ_word1-occ_word_beg)
        return result

    def p_score(self, word):
        """Not working, please ignore"""
        num = int(self.occurrences[self.vocab == word]) - self.expectation_gaussian(word)
        std = math.sqrt(self.__variance_gaussian(word))
        print("difference :", num)
        print("std : ", std)
        print(word)
        return num / std

    def p_score_max(self, word):
        """Returns the p_score of considered word. As a reminder:
        the more p_score is little (big), the more the word has chances to be unexpectedly rare (present).
        See readme for more info."""
        num = int(self.occurrences[self.vocab == word]) - self.expectation_gaussian(word)
        std = math.sqrt(self.variance_gaussian_max(word))
        print("difference :", num)
        print("std : ", std)
        print(word)
        if std == 0:
            return None
        return num / std

    def p_scores(self):
        """Returns the list of words of the sequence. It is sorted by predicted rarity."""
        obs_words = [word for word in self.vocab if len(word) == self.seq_len]
        p_scores = [(self.p_score_max(word), word) for word in obs_words if self.p_score_max(word) is not None]
        p_scores.sort()
        return p_scores


# Example
sequence_lenght = 6
with open("D:\Pierre\Git\MOPSI\MOPSI\E_coli.txt", 'r') as f:
    seq_complete = f.read().splitlines()
seq_complete = ''.join(seq_complete)
seq_complete = [seq_complete]

count = Counter_m(sequence_lenght)
count.learn(seq_complete)
p_sc = count.p_scores()
pv = [ps[0] for ps in p_sc]

# Histogram of p_scores for E_Coli DNA sequence
plt.hist(pv, bins='auto', normed=True, color='blue')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.show()
