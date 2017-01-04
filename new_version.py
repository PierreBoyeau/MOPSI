import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
import matplotlib.pyplot as plt


"""Cette méthode est la version généralisée de la méthode 6.2 du livre Méthodes Aléatoires"""
# --- Constants
alphabet = ['A', 'T', 'C', 'G']
markov_length = 4
sequence_lenght = 6
# markov_length = 4
# sequence_lenght = 6
assert(sequence_lenght >= markov_length-2)

# --- DNA samples examples
sequence_test = ["ATTCGCTGATTTGCCTCCGGCGGCTTTACTTTCTTTGCTGAATACCTGCGCGACAAGCTGCGCCTGACCGCCGTGGTAGCCGCCAGGCTTAACGCAACTGGTGCTCAAGCTGGAAACGCTGGGCTGGAAAGTGGCGAGAA"]

sequenceADN = ["AAATCTGCCGCTGATGCCAGGCTTAACGCAACTGGTGCTCAAGCTGGAAACGCTGGGCTGGAAAGTGGCGA" \
           "TTGCCTCCGGCGGCTTTACTTTCTTTGCTGAATACCTGCGCGACAAGCTGCGCCTGACCGCCGTGGTAGCC" \
           "AATGAACTGGAGATCATGGACGGTAAATTTACCGGCAATGTGATCGGCGACATCGTAGACGCGCAGTACAA" \
           "AGCGAAAACTCTGACTCGCCTCGCGCAGGAGTATGAAATCCCGCTGGCGCAGACCGTGGCGATTGGCGATG" \
           "GAGCCAATGACCTGCCGATGATCAAAGCGGCAGGGCTGGGGATTGCCTACCATGCCAAGCCAAAAGTGAAT" \
           "GAAAAGGCGGAAGTCACCATCCGTCACGCTGACCTGATGGGGGTATTCTGCATCCTCTCAGGCAGCCTGA" \
           "ATCAGAAGTAATTGCTCGCCCGCCATCCTGCGGGCGGCACAGCATTAACGAGGTACACCGTGGCAAAAGCT" \
           "CCAAAACGCGCCTTTGTTTGTAATGAATGCGGGGCCGATTATCCGCGCTGGCAGGGGC" \
           "GTGCAGTGCCTGTCATGCCTGGAACACCATCACCGAGGTGCGTCTTGCTGCGTCGCCA" \
          "ATGGTGGCGCGTAACGAGCGTCTCAGCGGCTATGCCGGTAGCGCCGGGGTGGCAAA" \
           "AGTCCAGAAACTCTCCGATATCAGCCTTGAAGAGCTGCCGCGTTTTTCCA" \
           "CCGGATTTAAAGAGTTCGACCGCGTACTAGGAACGCTGTGCAAACTGGCCCAGCA" \
           "GATGAAAACGCTGTATGTCACCGGCGAAGAGTCGCTGCAACAGGTGGCAATGCGCGCTCATCGCCTTGGC" \
           "CTGCCGACTGACAATCTCAATATGTTGTCGGAAACCAGCATCGAACAGATCTGCCTGATTGCCGAAGAAGAGCAACCG" \
           "AAGCTGATGGTAATTGACTCGATCCAGGTGATGCATATGGCGGATGTACAGTCATCGCCTGG" \
           "TAGCGTGGCGCAGGTGCGTGAAACGGCGGCTTATTTGACACGCTTCGCCAAAACGCGCGGTGTGGC"]

with open("D:\Pierre\Git\MOPSI\MOPSI\E_coli.txt", 'r') as f:
    seq_complete = f.read().splitlines()
seq_complete = ''.join(seq_complete)
seq_complete = [seq_complete]


# --- Building occurrences table
def make_chunks(s, m_len=markov_length, seq_len=sequence_lenght):
    return [s[i:i+width] for width in {m_len, m_len+1, seq_len}
            for i in range(len(s)-width+1)]

vectorizer = CountVectorizer(tokenizer=make_chunks, lowercase=False)
occurrences = vectorizer.fit_transform(seq_complete)
occurrences = occurrences.toarray()
occurrences = occurrences[0]
vocab = vectorizer.get_feature_names()
vocab = np.array(vocab)


# Compute expected value using Gaussian approximation
def expectation_gaussian(word, seq_len=sequence_lenght, m_len=markov_length):
    numerator = 1
    for j in range(0, seq_len-m_len):
        word1 = word[j:j+m_len+1]
        numerator *= int(occurrences[vocab == word1])
    denom = 1
    for j in range(1, seq_len-m_len):
        word1 = word[j:j+m_len]
        denom *= int(occurrences[vocab == word1])
    return numerator/denom  # Expect Python 3.X versions


# Compute variance value using Gaussian approximation
# #TODO tester
def variance_gaussian(word, seq_len=sequence_lenght, m_len=markov_length):
    expect = expectation_gaussian(word, seq_len, m_len)
    variance = expect

    # Computation of the independant term
    indep_cst = 0
    for d in range(0, sequence_lenght-m_len-1):
        if word[d] == word[sequence_lenght-1]:
            word_repeted = ''.join(word[:d]+word[:sequence_lenght])
            new_len = len(word_repeted)
            indep_cst += expectation_gaussian(word_repeted, new_len, m_len)
    variance += 2*indep_cst

    # Computation of the quadratic term
    quad_cst = 0
    word_vector = CountVectorizer(tokenizer=make_chunks, lowercase=False)
    word_occ = word_vector.fit_transform([word])
    word_occ = word_occ.toarray()
    word_occ = word_occ[0]
    word_vocab = word_vector.get_feature_names()
    word_vocab = np.array(word_vocab)
    word_vocab = [word for word in word_vocab if len(word) != seq_len]
    for subword in word_vocab:
        if len(subword) == m_len:
            apparitions = [int(word_occ[word_vocab == subword+k]) for k in alphabet if (subword+k in word_vocab)]
            num = sum(apparitions)
            quad_cst += (num**2)/int(occurrences[vocab == subword])
        elif len(subword) == m_len+1:
            quad_cst -= (int(word_occ[word_vocab == subword])**2)/int(occurrences[vocab == subword])

    apparitions = [int(word_occ[word_vocab == word[0:m_len-1]+k]) for k in alphabet if (word[0:m_len-1]+k in word_vocab)]
    num = sum(apparitions)
    quad_cst += (1 - 2*num)/int(occurrences[vocab == word])
    variance += (expect*expect)*quad_cst

    return variance


def variance_gaussian_max(word, seq_len=sequence_lenght, m_len=markov_length):
    assert(seq_len == m_len + 2)
    expect = expectation_gaussian(word, seq_len, m_len)
    occ_word1 = int(occurrences[vocab == word[1:seq_len - 1]])
    occ_word_end = int(occurrences[vocab == word[1:seq_len]])
    occ_word_beg = int(occurrences[vocab == word[0:seq_len - 1]])
    result = expect/(occ_word1**2)*(occ_word1-occ_word_end)*(occ_word1-occ_word_beg)
    return result


def p_score(word):
    num = int(occurrences[vocab == word]) - expectation_gaussian(word)
    std = math.sqrt(variance_gaussian(word))
    print("difference :", num)
    print("std : ", std)
    print(word)
    return num / std


def p_score_max(word):
    num = int(occurrences[vocab == word]) - expectation_gaussian(word)
    std = math.sqrt(variance_gaussian_max(word))
    print("difference :", num)
    print("std : ", std)
    print(word)
    return num / std

# # --- Script
obs_words = [word for word in vocab if len(word) == sequence_lenght]
