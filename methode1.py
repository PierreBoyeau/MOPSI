import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import math
import matplotlib.pyplot as plt


"""Cette méthode est la version correspondante a 6.1 de Methodes Aleatoires"""
# --- Constants
alphabet = ['A', 'T', 'C', 'G']
markov_length = 1 # Please do not modify value.
sequence_lenght = 6

assert(sequence_lenght >= markov_length-2)

# --- DNA samples examples
# sequence_test = ["ATTCGCTGATTTGCCTCCGGCGGCTTTACTTTCTTTGCTGAATACCTGCGCGACAAGCTGCGCCTGACCGCCGTGGTAGCCGCCAGGCTTAACGCAAC \
#                   TGGTGCTCAAGCTGGAAACGCTGGGCTGGAAAGTGGCGAGAA"]
#
# sequenceADN = ["AAATCTGCCGCTGATGCCAGGCTTAACGCAACTGGTGCTCAAGCTGGAAACGCTGGGCTGGAAAGTGGCGA" \
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
#            "TAGCGTGGCGCAGGTGCGTGAAACGGCGGCTTATTTGACACGCTTCGCCAAAACGCGCGGTGTGGC"]

with open("D:\Pierre\Git\MOPSI\MOPSI\E_coli.txt", 'r') as f:
    seq_complete = f.read().splitlines()
seq_complete = ''.join(seq_complete)
seq_complete = [seq_complete]


# --- Building occurrences table
def make_chunks(s):
    return [s[i:i+width] for width in {1, 2, sequence_lenght-1, sequence_lenght}
            for i in range(len(s)-width+1)]


vectorizer = CountVectorizer(tokenizer=make_chunks, lowercase=False)
occurrences = vectorizer.fit_transform(seq_complete)
occurrences = occurrences.toarray()
occurrences = occurrences[0]
vocab = vectorizer.get_feature_names()
vocab = np.array(vocab)

# ---
# ---Useful functions


def occ(word):
    return int(occurrences[vocab == word])


# Attention par rapport a l'ouvrage Modeles aleatoires on fait tout de suite la simplification par sqrt(N)
def expectation(word):
    """Estimator computed thanks to the Ergodic Theorem"""
    return occ(word[:-1])*occ(word[-2:]) / occ(word[-2])


def variance(word):
    """Estimation of variance of expected occurences for given word"""
    frac = occ(word)
    term1 = 1 - occ(word[:-1])/occ(word[-2])
    term2 = 1 - occ(word[-2:])/occ(word[-2])
    return frac * term1 * term2


def p_score(word):
    return (occ(word) - expectation(word)) / math.sqrt(variance(word))


# --- Script
obs_words = [word for word in vocab if len(word) == sequence_lenght]
p_scores = [(p_score(word), word) for word in obs_words]
print(min(p_scores))