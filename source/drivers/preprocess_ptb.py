import sys
sys.path.insert(0, "source/utilities")
import corpus_utils as utils

corpora = [
    {
        "path": "data/ptb/ptb.train.txt",
        "delimiter": '\n',
        "validate_frac": 0,
        "word_tokenization": utils.SPLIT_ON_SPACES
    },
    {
        "path": "data/ptb/ptb.valid.txt",
        "delimiter": '\n',
        "validate_frac": 1,
        "word_tokenization": utils.SPLIT_ON_SPACES
    }
]
utils.consolidate(corpora, "data/preprocessed/ptb.hdf5")
