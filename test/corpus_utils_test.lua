require "source/utilities/corpus_utils.lua"

data = load_hdf5("data/preprocessed/ptb.hdf5")
batch_data = batch_documents(20, 100, data["train"])
