<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      04/28/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

A collection of experiments for sequence learning using RNN's.

# TODO

- Data preprocessing.
- Revised version of `run_model.lua`.
  - Fix the bug when using `replace` but the file does not exist.
- Add the sudden decay strategy used by Zaremba's code as a schedule to `sopt`.

# Things to Try

- Shuffling corpura that consist of independent documents. The HDF5 file will
make it easy to do this.
- Mixing corpuses from different authors for sequence prediction.

- Performance of different optimization algorithms.
- Performance of various types of LSTM units (e.g. with and without peephole
connections).
- Performance of SCRNN.

- Using convolution with RNN's.
- Using NCE instead of softmax.
