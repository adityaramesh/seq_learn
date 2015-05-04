dofile "source/utilities/corpus_utils.lua"

data = load_hdf5("data/preprocessed/ptb.hdf5")
-- TODO test
batch_data = batch_documents(20, 200, data["train"])
-- TODO test
