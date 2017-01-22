import numpy as np


seq_len = 1000
mu = [0.1, 0.4, 0.4, 0.1]
trans_mat = [[0.1, 0.5, 0.3, 0.1],
             [0.1, 0.4, 0.4, 0.1],
             [0.05, 0.45, 0.45, 0.05],
             [0.2, 0.3, 0.3, 0.2]]
alphabet = ['A', 'T', 'C', 'G']

# Building cdf
cdf_mu = [mu[0]]
for i in range(1, 4):
    cdf_mu = cdf_mu + [cdf_mu[-1] + mu[i]]
cdf_mat = np.zeros((4, 4))
for line in range(0, 4):
    cdf_mat[line, 0] = trans_mat[line][0]
    for col in range(1, 4):
        cdf_mat[line, col] = cdf_mat[line, col-1] + trans_mat[line][col]
seq = ""


def choice(value, cdf, previous_index):
    for index in range(0, 4):
        if value <= cdf[previous_index][index]:
            return index

rands = np.random.random(seq_len)
i = 0
while rands[0] > cdf_mu[i]:
    i += 1
last_i = i
seq += alphabet[i]

for j in range(seq_len):
    index = choice(rands[j], cdf_mat, last_i)
    last_i = index
    seq += alphabet[last_i]
