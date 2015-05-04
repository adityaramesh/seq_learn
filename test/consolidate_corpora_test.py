import h5py
import numpy as np

output_fp = "data/preprocessed/ptb.hdf5"
data = h5py.File(output_fp, "r")
train_data = data["train"]["ptb.train.txt"]

vocab = {}
cur_word = bytearray()
cur_index = 0

for i in data["vocab"]:
    if i != 0:
        cur_word.append(i)
    else:
        vocab[cur_index] = cur_word.decode("utf-8")
        cur_word = bytearray()
        cur_index = cur_index + 1

line = [""]
lines = set()

for indices, length in zip(train_data["contents"], train_data["lengths"]):
    for i in range(length):
        line.append(vocab[indices[i]])
    line.append("")
    lines.add(str.join(" ", line))
    line = [""]

train_fp = "data/ptb/ptb.train.txt"
train_file = open(train_fp, "r")
orig_lines = [line for line in train_file.readlines()]

for line in orig_lines:
    if not line.strip("\n") in lines:
        print(line)
