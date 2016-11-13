import math
import itertools
import numpy as np

"""This program allows to compute the p-value
of different words in a NDA sequence in order to determine abnormally
rare or frequent words.
Please modify the word length"""


dna_sequence = ''
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

    def build_occurrences_method1(self, sequence):
        """Builds the occurrences tables for h=1, 2, h-1 and h"""
        self.sequence_length = len(sequence)
        lengths_already_processed = []
        for word_length in [1, 2, self.length-1, self.length]:
            if word_length not in lengths_already_processed:
                for position in range(word_length, self.sequence_length+1, 1):
                    observed_word = sequence[position-word_length:position]
                    self[observed_word] += 1
                    print(observed_word)
                lengths_already_processed.append(word_length)

    def get_zetan_obs(self):
        """Has to be tested and optimized if necessary"""
        zetan = np.empty(4 ** self.length)
        for word_tuple in itertools.product(E, repeat=self.length):
            word = ''.join(word_tuple)
            size, key = get_key(word)
            first_estimator = self[word]
            second_estimator = self[word[:-1]]*self[word[-2]+word[-1]]/self[word[-2]]
            zetan[key] = 1/math.sqrt(self.sequence_length)*(first_estimator - second_estimator)
        return zetan

    def get_zn_obs(self):
        """Has to be tested and optimized if necessary"""
        zn = np.empty(4 ** self.length)
        for word_tuple in itertools.product(E, repeat=self.length):
            word = ''.join(word_tuple)
            size, key = get_key(word)
            first_estimator = self[word]
            second_estimator = self[word[:-1]]*self[word[-2]+word[-1]]/self[word[-2]]
            zeta = 1/math.sqrt(self.sequence_length)*(first_estimator - second_estimator)
            sigma = math.sqrt(first_estimator/self.sequence_length
                              * (1 - (self[word[:-1]]/self[word[-2]]))
                              * (1 - (second_estimator/self.sequence_length))
                              )
            zn[key] = zeta / sigma
        return zn


# Step 1 : create table of words occurrences for length in range(1,h+1)


A = Occurrences(3)
A.build_occurrences_method1('ATCGATCGATCG')
print(A.table[0])
print(A.table[1])
print(A["GA"])