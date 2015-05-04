#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Takes a dictionary describing a corpora and puts the contents into a single file
that makes preprocessing and training easier.
"""

import sys
import os.path
import h5py

import re
import math
import numpy as np

# Used to control corpus granularity.
WORD = 0
CHARACTER = 1

# Used to control word tokenization.
DEFAULT = 0
SPLIT_ON_SPACES = 1

class LazyGroupManager:
    """
    Prior to processing the corpora, it is not possible to tell whether we will
    have to create the top-level groups `train` and `validate`. Creating these
    groups initially and removing them later if they are not used is not a good
    option, because this will result in fragmentation within the file. Instead,
    we create the groups lazily using this class.
    """

    def __init__(self, output_file):
        self.output_file = output_file
        self.train_group = None
        self.validate_group = None

    def get_train_group(self):
        if self.train_group == None:
            self.train_group = self.output_file.create_group("/train")
        return self.train_group

    def get_validate_group(self):
        if self.validate_group == None:
            self.validate_group = self.output_file.create_group("/validate")
        return self.validate_group

def corpus_iter_words(corpus):
    """
    Iterates over a corpus at the word level.
    """

    corpus_file = open(corpus["path"], mode='r', encoding=corpus["encoding"])

    if corpus["word_tokenization"] == DEFAULT:
        """
        This is the regex that we use to perform the default word tokenization.
        Use http://regexper.com to visualize it.

        Examples of words that we parse in their entirety:
        - M&Ms
        - 123.45-point
        - magnitude-7.0
        - they're

        Examples of words that are broken up due to ambiguities that are
        difficult to resolve:
        - 'tis (becomes ["'", "tis"])
        - U.S. (becomes ["U", ".", "S", "."])
        """
        regex = r"(?:\d+(?:\.\d+)*|[\w<>]+)(?:[-'&]?(?:\d+(?:\.\d+)*|[\w<>]+))*|--|[^\w\s]"
        delimiter = corpus["delimiter"]
        if delimiter != "":
            regex = regex + "|" + delimiter

        text = corpus_file.read()
        for token in re.findall(regex, text, re.UNICODE):
            yield token
    else:
        for token in corpus_file.read().split(' '):
            yield token

def corpus_iter_chars(corpus):
    """
    Iterates over a corpus at the character level. Things are complicated by the
    fact that we have to scan for multicharacter tokens. Note that if a
    delimiter was provided, then it was added to `multichar_tokens` earlier. So
    we do not have to check for the delimiter explicitly.
    """

    corpus_file = open(corpus["path"], mode='r', encoding=corpus["encoding"])
    multichar_tokens = corpus["multichar_tokens"]

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
    if granularity == WORD:
        return corpus_iter_words(corpus)
    else:
        return corpus_iter_chars(corpus)

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

    frac = corpus["validate_frac"]
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
            group.create_dataset("contents", data=docs_array[start_doc:end_doc])
            group.create_dataset("lengths", data=doc_lengths_array[start_doc:end_doc])

        first_validate_doc = len(docs) - validate_docs
        if first_validate_doc != 0:
            write_data(manager.get_train_group(), 0, first_validate_doc)
        if validate_docs != 0:
            write_data(manager.get_validate_group(), first_validate_doc, len(docs))

    print("'{}': {} documents processed.".format( \
        os.path.basename(corpus["path"]), len(docs)))

def consolidate(corpora, output_fp, granularity=WORD):
    """
    Consolidates the contents of `corpora` into `output_fp`.

    `granularity` determines whether tokens in the corpora are defined to be
    words or characters. Allowed values are `WORD` and `CHARACTER`.

    `corpora` must be an array of dictionaries, each of which must have the
    following keys:
    - `path`: The path to the file containing the corpus.
    - `encoding`: The default value is UTF-8.
    - `delimiter`: If present, assumes that the file consists of a list of
      **independent** documents separated by `delimiter`. This implies that
      context is not carried over from document to document, so that it is safe
      to shuffle the order of the documents. `delimiter` can consist of more
      than one character, even if `granularity == CHARACTER`.
    - `validate_frac`: Only allowed for files that consist of more than one
      document (so `delimiter` must be specified). The validation set is taken
      to be the last `n` documents in the file, where `n =
      math.ceil(validate_frac * total_docs)`. The default value is `0`.
    - `word_tokenization`: Only allowed if `granularity == WORD`. Allowed values
      are `DEFAULT` (which is the default value) and `SPLIT_ON_SPACES`.
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

    invalid_key = False
    valid_keys = ["path", "encoding", "delimiter", "validate_frac", \
        "word_tokenization", "multichar_tokens"]

    for corpus in corpora:
        for key in corpus:
            if not key in valid_keys:
                print("Invalid key '{}'.".format(key), file=sys.stderr)
                invalid_key = True

        if not "encoding" in corpus:
            corpus["encoding"] = "utf-8"
        if not "delimiter" in corpus:
            corpus["delimiter"] = ""
        if not "validate_frac" in corpus:
            corpus["validate_frac"] = 0

        if "word_tokenization" in corpus:
            if granularity == CHARACTER:
                raise RuntimeError("The 'word_tokenization' key should only " \
                    "be present when the corpora are parsed at word-level " \
                    "granularity.")
        else:
            corpus["word_tokenization"] = DEFAULT

        if "multichar_tokens" in corpus:
            if granularity != CHARACTER:
                raise RuntimeError("The 'multichar_tokens' key should only be " \
                    "present when the corpora are parsed at character-level " \
                    "granularity.")

            delim = corpus["delimiter"]
            if delim != "" and not delim in corpus["multichar_tokens"]:
                corpus["multichar_tokens"].append(delim)
        else:
            corpus["multichar_tokens"] = []

    if invalid_key:
        raise RuntimeError("Unknown keys found.")

    vocab = {}
    output_file = h5py.File(output_fp, "w")
    manager = LazyGroupManager(output_file)

    for corpus in corpora:
        process_corpus(corpus, vocab, manager, granularity)

    # For compatibility with Torch, we have to serialize the strings as byte
    # arrays. The cleanest way to serialize the strings is to join their UTF-8
    # decoded byte arrays, using the null byte as the delimiter. We do not use
    # this scheme for storing multi-document corpora, because it would make them
    # difficult to shuffle.
    byte_array = []
    for token in sorted(vocab, key=vocab.get):
        byte_array.extend(bytearray(token, encoding="utf-8"))
        byte_array.append(0)
    output_file.create_dataset("vocab", data=byte_array)
