--
-- This file is used to train RNN's using window-based BPTT. Window-based BPTT
-- does not use truncation. Instead, the documents are divided into batch_size
-- groups, and the documents within the same group are packed into a row:
--
-- 	Row 1: [***********] [*******] [*********************]
-- 	Row 2: [***] [*********] [************] [********]
-- 	Row 3: [*******] [********] [***] [***********] [*****]
--
-- BPTT is performed for all documents that fit in their entirety within the
-- current window. Context is managed carefully so that it does not carry over
-- between separate documents.
--

require "torch"
require "xlua"
require "source/utilities/lstm_rnn"
require "source/utilities/corpus_utils"
require "source/utilities/serialization"

local function get_model_info(params)
	--
end

local function get_train_info(params)
	--
end

local function do_train_epoch(data, context, paths, info)
	--
end

local function do_valid_epoch(data, context, paths, info)
	--
end

function run(params)
	print("Loading data.")
	local data = load_hdf5("data/preprocessed/ptb.hdf5")
	print("Preprocessing validation data.")
	local valid_data = batch_documents(params.batch_size, data.valid)

	local do_train, _, paths, info = restore(
		function() return get_model_info(params) end,
		function() return get_train_info(params) end
	)
	info.model:restore()

	local context = {}
	context.params, context.param_grads = info.model.network:getParameters()
	context.docs_processed = torch.LongTensor(params.batch_size)
	context.tokens_processed = torch.LongTensor(params.batch_size)

	if do_train then
		while true do
			do_train_epoch(data, context, paths, info)
			do_valid_epoch(valid_data, context, paths, info)
		end
	else
		do_valid_epoch(valid_data, context, paths, info)
	end
end
