import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pylab as P

"""This program allows to compute the p-value
of different words in a NDA sequence in order to determine abnormally
rare or frequent words.
Please modify the word length"""


E = ['A', 'T', 'C', 'G']
DICT_E = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3
}


def get_key(str):
    size = len(str) - 1
    key, power = 0, 1
    for letter in str:
        key += power * DICT_E[letter]
        power *= 4
    return size, key


class Occurrences:
    def __init__(self, length):
        """
        length : length of words we want to study
        table : list of occurences of all words than have sizes from 1 to length.
            For instance, self.table[1] returns the occurrences of words of size 2.
        sequence_length : length of studied sequence
        """
        assert (length >= 3)
        self.length = length
        self.sequence_length = 0
        self.table = []
        number_of_words = 4
        for iterable in range(length):
            self.table.append([0]*number_of_words)
            number_of_words *= 4

    def __setitem__(self, str, value):
        size, key = get_key(str)
        self.table[size][key] = value

    def __getitem__(self, str):
        size, key = get_key(str)
        return self.table[size][key]


class Occurrences1(Occurrences):
    def __init__(self, length, sequence):
        Occurrences.__init__(self, length)
        self.zn = np.empty(4 ** self.length)
        self.pn = np.empty(4 ** self.length)
        self.build_occurrences_method1(sequence)
        self.get_zn_method1()
        self.get_proba_method1()

    def build_occurrences_method1(self, sequence):
        """Builds the occurrences tables for h=1, 2, h-1 and h"""
        self.sequence_length = len(sequence)
        lengths_already_processed = []
        for word_length in [1, 2, self.length-1, self.length]:
            if word_length not in lengths_already_processed:
                for position in range(word_length, self.sequence_length+1, 1):
                    observed_word = sequence[position-word_length:position]
                    self[observed_word] += 1
                lengths_already_processed.append(word_length)

    def get_zn_method1(self):
        """Has to be tested and optimized if necessary
        function provoked error when a word has no occurrence.
        solution found : test whether first_estimator is equal to 0 (it's an it, thus valid comparison)"""
        for word_tuple in itertools.product(E, repeat=self.length):
            word = ''.join(word_tuple)
            size, key = get_key(word)
            first_estimator = self[word]
            second_estimator = np.true_divide(self[word[:-1]]*self[word[-2]+word[-1]], self[word[-2]])
            if first_estimator != 0:
                zeta = (first_estimator - second_estimator)/math.sqrt(self.sequence_length)

                sigma = math.sqrt(np.true_divide(first_estimator, self.sequence_length)
                                  * (1 - (self[word[:-1]]/self[word[-2]]))
                                  * (1 - (second_estimator/self.sequence_length))
                                  )
                self.zn[key] = np.true_divide(zeta, sigma)
            else:
                self.zn[key] = np.nan

    def get_proba_method1(self):
        """Error may come from the fact that when must check wheter word has occurence ?
        + maybe approximation error"""
        cdf = scipy.stats.norm.cdf
        for i in range(4 ** self.length):
            if self.zn[i] != np.nan:
                self.pn[i] = 1 - cdf(self.zn[i])
            else:
                self.pn[i] = np.nan


sequence = random_sequence(8000)
a = Occurrences1(4, sequence)
b = a.zn[~np.isnan(a.zn)]

hist = np.histogram(b, density=True)
print(hist)
plt.hist2d(hist[0], hist[1])
plt.show()

#-------------------------------
#-------------------------------
#-------------------------------
#-------------------------------ETUDE D UN BRIN D ADN
# sequenceADN = "AAATCTGCCGCTGATGCCAGGCTTAACGCAACTGGTGCTCAAGCTGGAAACGCTGGGCTGGAAAGTGGCGA" \
#            "TTGCCTCCGGCGGCTTTACTTTCTTTGCTGAATACCTGCGCGACAAGCTGCGCCTGACCGCCGTGGTAGCC" \
#            "AATGAACTGGAGATCATGGACGGTAAATTTACCGGCAATGTGATCGGCGACATCGTAGACGCGCAGTACAA" \
#            "AGCGAAAACTCTGACTCGCCTCGCGCAGGAGTATGAAATCCCGCTGGCGCAGACCGTGGCGATTGGCGATG" \
#            "GAGCCAATGACCTGCCGATGATCAAAGCGGCAGGGCTGGGGATTGCCTACCATGCCAAGCCAAAAGTGAAT" \
#            "GAAAAGGCGGAAGTCACCATCCGTCACGCTGACCTGATGGGGGTATTCTGCATCCTCTCAGGCAGCCTGA" \
#            "ATCAGAAGTAATTGCTCGCCCGCCATCCTGCGGGCGGCACAGCATTAACGAGGTACACCGTGGCAAAAGCT" \
#            "CCAAAACGCGCCTTTGTTTGTAATGAATGCGGGGCCGATTATCCGCGCTGGCAGGGGC" \
#            "GTGCAGTGCCTGTCATGCCTGGAACACCATCACCGAGGTGCGTCTTGCTGCGTCGCCA" \
#            "ATGGTGGCGCGTAACGAGCGTCTCAGCGGCTATGCCGGTAGCGCCGGGGTGGCAAA" \
#            "AGTCCAGAAACTCTCCGATATCAGCCTTGAAGAGCTGCCGCGTTTTTCCA" \
#            "CCGGATTTAAAGAGTTCGACCGCGTACTAGGAACGCTGTGCAAACTGGCCCAGCA" \
#            "GATGAAAACGCTGTATGTCACCGGCGAAGAGTCGCTGCAACAGGTGGCAATGCGCGCTCATCGCCTTGGC" \
#            "CTGCCGACTGACAATCTCAATATGTTGTCGGAAACCAGCATCGAACAGATCTGCCTGATTGCCGAAGAAGAGCAACCG" \
#            "AAGCTGATGGTAATTGACTCGATCCAGGTGATGCATATGGCGGATGTACAGTCATCGCCTGG" \
#            "TAGCGTGGCGCAGGTGCGTGAAACGGCGGCTTATTTGACACGCTTCGCCAAAACGCGCGGTGTGGC"
#
# a = Occurrences1(6, sequenceADN)
# plt.hist(a.zn)
# plt.show()
#