#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Takes a dictionary describing a corpora and puts the contents into a single file
on which a model can be easily trained.
"""

from enum import Enum

import sys
import os.path
import h5py

import math
import numpy as np

class Granularity(Enum):
    word = 1,
    character = 2

class LazyGroupManager:
    def __init__(self, output_file):
        self.output_file = output_file
        self.train_group = None
        self.validate_group = None

    def get_train_group(self):
        if self.train_group == None:
            self.train_group = self.output_file.create_group("/train")

    def get_validate_group(self):
        if self.validate_group == None:
            self.validate_group = self.output_file.create_group("/validate")

def corpus_iter_words(corpus_file):
    """
    Iterates over a corpus at the word level.
    """
    for word in corpus_file.read().split(' '):
        yield word

def corpus_iter_chars(corpus_file, multichar_tokens):
    """
    Iterates over a corpus at the character level. Things are complicated by the
    fact that we have to scan for multicharacter tokens. Note that if a
    delimiter was provided, then it was added to `multichar_tokens` earlier. So
    we do not have to check for the delimiter explicitly.
    """

    # Used to keep track of previous characters that may be the start of a
    # multicharacter token.
    buf = ""

    while True:
        c = corpus_file.read(1)
        if c == '':
            for char in buf:
                yield char
            raise StopIteration()

        buf += c
        for token in multichar_tokens:
            if token == buf:
                buf = ""
                yield token
                continue

        # Drain as many characters from the front of `buf` as possible.
        while True:
            possible_match = False
            for token in multichar_tokens:
                if token.startswith(buf):
                    possible_match = True
                    break

            if possible_match:
                break
            else:
                yield buf[0]
                buf = buf[1:]
                if len(buf) == 0:
                    break

def corpus_iter(corpus, granularity):
    with open(corpus["path"], mode='r', encoding=corpus["encoding"]) as f:
        if granularity == Granularity.word:
            return corpus_iter_words(f)
        else:
            return corpus_iter_chars(f, corpus["multichar_tokens"])

def parse_corpus(corpus, vocab, granularity):
    doc = []
    docs = []
    doc_length = 0
    max_doc_len = 0
    doc_lengths = []

    for token in corpus_iter(corpus, granularity):
        if token == corpus["delimiter"]:
            docs.append(doc)
            doc_lengths.append(doc_length)
            max_doc_len = max(max_doc_len, doc_length)
            doc = []
            doc_length = 0
            continue

        doc_length = doc_length + 1
        if token in vocab:
            doc.append(vocab[token])
        else:
            doc.append(len(vocab))
            vocab[token] = len(vocab)

    return docs, doc_lengths, max_doc_len

def process_corpus(corpus, vocab, manager, granularity):
    docs, doc_lengths, max_doc_len = parse_corpus(corpus, vocab, granularity)

    frac = corpus["validation_frac"]
    validate_docs = math.ceil(frac * len(docs))
    if frac != 1 and validate_docs == len(docs):
        print("Warning: requested fraction of validation documents rounds up " \
            "to total number of documents in corpus '{}'.".format(corpus.path), \
            file=sys.stderr)

    basename = os.path.basename(corpus["path"])

    if len(docs) == 1:
        docs_array = np.array(docs[0], dtype=np.uint32)

        def write_data(parent):
            group = parent.create_group(basename)
            group.create_dataset("contents", data=docs_array)

        if validate_docs == 0:
            write_data(manager.get_train_group())
        else:
            write_data(manager.get_validate_group())
    else:
        perm = np.random.permutation(len(docs))
        docs_array = np.zeros(shape=(len(docs), max_doc_len), dtype=np.uint32)
        doc_lengths_array = np.array(doc_lengths, dtype=np.uint32)

        for i, doc in enumerate(docs):
            for j, index in enumerate(doc):
                docs_array[perm[i]][j] = index

        def write_data(parent, start_doc, end_doc):
            group = parent.create_group(basename)
            group.create_dataset("contents", data=\
                docs_array[start_doc:end_doc])
            group.create_dataset("lengths", data=\
                doc_lengths_array[start_doc:end_doc])

        first_validate_doc = len(docs) - validate_docs
        if first_validate_doc != 0:
            write_data(manager.get_train_group(), 0, first_validate_doc)
        if validate_docs != 0:
            write_data(manager.get_validate_group(), first_validate_doc, \
                len(docs))

def consolidate(corpora, output_fp, granularity=Granularity.word):
    """
    Consolidates the contents of `corpora` into `output_fp`.

    `granularity` determines whether tokens in the corpora are defined to be
    words or characters. Allowed values are `Granularity.word` and
    `Granularity.character`.

    `corpora` must be an array of dictionaries, each of which must have the
    following keys:
    - `path`: The path to the file containing the corpus.
    - `encoding`: The default value is UTF-8.
    - `delimiter`: If present, assumes that the file consists of a list of
      **independent** documents separated by `delimiter`. This implies that
      context is not carried over from document to document, so that it is safe
      to shuffle the order of the documents. `delimiter` can consist of more
      than one character, even if `granularity == Granularity.character`.
    - `validation_frac`: Only allowed for files that consist of more than one
      document (so `delimiter` must be specified). The validation set is taken
      to be the last `n` documents in the file, where `n =
      math.ceil(validation_frac * total_docs)`. The default value is `0`.
    - `multichar_tokens`: Can only be present if `granularity ==
      Granularity.character`. In this case, the value should be an array of
      strings. Each string in the array describes a word that should be treated
      as a single token. I.e., the words in the array are not broken up into
      individual characters during parsing.

    `output_fp` is the path to the output HDF5 file for the preprocessed
    corpora. The following will be the top-level structures in the file:
    - "/vocab": Contains a list of the unique tokens occurring in `corpora`. The
      position in which a token occurs corresponds to its (zero-based) index.
    - "/train/": Contains the training sets. Won't be created if all corpora are
      used solely for validation.
    - "/validation/": Contains the validation sets. Won't be created if all
      corpora are used solely for training.

    Each top-level group contains one or more subgroups. The name of each
    subgroup is the basename of the file from which it is derived. If the file
    consists of a single document, then the subgroup will contain a single
    dataset with the name `contents`. This dataset will be a 1D array of 32-bit
    unsigned indices. If the file consists of more than one document, then
    `contents` will be a 2D array of 32-bit unsigned indices.  Each row
    corresponds to one document. In addition, the subgroup will contain a
    dataset called `lengths`. This will be a 1D array of 32-bit unsigned
    indices; each index indicates the length in tokens of the corresponding
    document in `contents`.
    """

    if os.path.isfile(output_fp):
        raise RuntimeError("The file '{}' already exists. Remove it first if " \
            "you wish to replace it.".format(output_fp))
    if len(corpora) == 0:
        raise RuntimeError("The list of corpora is empty.")

    for corpus in corpora:
        if not "encoding" in corpus:
            corpus["encoding"] = "utf-8"
        if not "delimiter" in corpus:
            corpus["delimiter"] = ""
        if not "validation_frac" in corpus:
            corpus["validation_frac"] = 0

        if "multichar_tokens" in corpus:
            if granularity != Granularity.character:
                raise RuntimeError("The 'multichar_tokens' key should only be " \
                    "present when the corpora are parsed at character-level " \
                    "granularity.")

            delim = corpus["delimiter"]
            if delim != "" and not delim in corpus["multichar_tokens"]:
                corpus["multichar_tokens"].append(delim)
        else:
            corpus["multichar_tokens"] = []

    vocab = {}
    output_file = h5py.File(output_fp, "w")
    manager = LazyGroupManager(output_file)

    for corpus in corpora:
        process_corpus(corpus, vocab, manager, granularity)

    max_len = max(len(word) for word in vocab)
    vocab_array = np.empty(shape=(len(vocab)), dtype=('S', max_len))
    for token, index in vocab:
        vocab_array[index] = token
    output_file.create_dataset("vocab", data=vocab_array)
