<!--
  ** File Name: README.md
  ** Author:    Aditya Ramesh
  ** Date:      04/28/2015
  ** Contact:   _@adityaramesh.com
-->

# Overview

A collection of experiments for sequence learning using RNN's.

# TODO

- Add support for saving training and validation errors in a format that makes
it easy to graph progress using Python.
- Decouple the network and the criterion, and integrate RNNContext with
Zaremba's code.
- Get the window-based BPTT working.
  - Print out the mean and std of the number of documents processed per window.
  This should be used to compare against the hard cutoff BPTT strategy.
- Add the sudden decay strategy used by Zaremba's code as a schedule to `sopt`.

# Things to Try

- Idea for training an architecture to generate text in constrained formats
(e.g. sonnets, haikus, etc.). In these cases, we have lots of unstructured text
available, but relatively few examples of text in the constrained format.
  - First train an architecture as usual on a large corpus of text.
  - Then modify the softmax layer to incorporate the constraints of the desired
  text format. E.g. for haikus, fix the number of lines, syllables, etc.
  - Finally, train the architecture on the examples of text in the constrained
  format.

- Train a bidirectional RNN and use knowledge distillation to train a forward
RNN using the posterior probabilities.

- Shuffling corpura that consist of independent documents. The HDF5 file will
make it easy to do this.
- Mixing corpuses from different authors for sequence prediction.

- Performance of different optimization algorithms.
- Performance of various types of LSTM units (e.g. with and without peephole
connections).
- Performance of SCRNN.

- Using convolution with RNN's.
- Using NCE instead of softmax.
