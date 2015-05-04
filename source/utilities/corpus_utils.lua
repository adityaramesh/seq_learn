require 'hdf5'

--
-- Used to copy all of the training documents (or validation documents) from the
-- corpus in the file to the same arrays.
--
local function merge_documents(category, data)
	-- First, we accumulate some counts in order to determine the size of
	-- the arrays to allocate.
	local max_doc_len = 0
	local doc_count = 0

	for name, corpus in pairs(data[category]) do
		if corpus["contents"] == nil then
			error("\"contents\" key must be present for all corpora.")
		end

		if corpus["lengths"] ~= nil then
			local doc_len = corpus["contents"]:size(2)
			assert(doc_len > 0)

			-- The corpus consists of multiple documents.
			max_doc_len = math.max(max_doc_len, doc_len)
			doc_count = doc_count + corpus["contents"]:size(1)
		else
			local doc_len = corpus["contents"]:size(1)
			assert(doc_len > 0)

			-- The corpus consists of only one document.
			max_doc_len = math.max(max_doc_len, doc_len)
			doc_count = doc_count + 1
		end
	end

	-- Now we allocate the memory and perform the copies.
	local docs = torch.IntTensor(doc_count, max_doc_len):zero()
	local lengths = torch.IntTensor(doc_count)
	local cur_doc = 1

	for name, corpus in pairs(data[category]) do
		if corpus["lengths"] ~= nil then
			rows = corpus["contents"]:size(1)
			cols = corpus["contents"]:size(2)
			docs[{{cur_doc, cur_doc + rows - 1}, {1, cols}}] = corpus["contents"]
			lengths[{{cur_doc, cur_doc + rows - 1}}] = corpus["lengths"]
			cur_doc = cur_doc + rows
		else
			cols = corpus["contents"]:size(1)
			docs[{cur_doc, {1, cols}}] = corpus["contents"]
			lengths[cur_doc] = cols
			cur_doc = cur_doc + 1
		end
	end

	return docs, lengths
end

function load_hdf5(fn)
	local data = hdf5.open(fn):read():all()
	local train_docs, train_lengths = merge_documents("train", data)
	local validate_docs, validate_lengths = merge_documents("validate", data)

	local vocab = {}
	local cur_index = 0
	local cur_word = {}

	for i = 1, data["vocab"]:size(1) do
		if data["vocab"][i] == 0 then
			vocab[cur_index] = table.concat(cur_word, "")
			cur_word = {}
			cur_index = cur_index + 1
		else
			cur_word[#cur_word + 1] = string.char(data["vocab"][i])
		end
	end

	local new_data = {
		vocab = vocab,
		train = {
			documents = train_docs,
			lengths = train_lengths
		},
		validate = {
			documents = validate_docs,
			lengths = validate_lengths
		}
	}

	data = nil
	collectgarbage()
	return new_data
end

actions = {
	truncate_bptt = 0,
	end_document = 1
}

--
-- Used to put the training or validation documents returned by `load_hdf5` into
-- a format that can be fed into an RNN in batch mode.
--
function batch_documents(batch_size, max_bptt_len, data)
	local doc_count = data["lengths"]:size(1)
	assert(batch_size <= doc_count)

	--
	-- Suppose that `batch_size == 2`. Then the first component of the input
	-- to the network will come from documents 1 and 3, while the second
	-- component of the input will come from document 2.
	--
	-- Document 1: [*****************] (length 17)
	-- Document 2: [***********************] (length 23)
	-- Document 3: [*******] (length 7)
	--
	-- In batch mode, each ith component of the input to the network will be
	-- from the document whose (one-based) index minus one and modulo
	-- `batch_size` is `i - 1`. In order to determine the number of columns
	-- for the data array with `batch_size` rows, we need to compute `max_i
	-- L_i`, where `L_i` is the sum of the lengths of the documents whose
	-- index minus one and modulo `batch_size` is `i - 1`. In the example
	-- above, this value is 17 + 7 = 24.
	--
	local perm = torch.randperm(doc_count)
	local batch_lengths = torch.IntTensor(batch_size):zero()

	for i = 1, doc_count, batch_size do
		local batches = i + batch_size - 1 > doc_count and
			doc_count % batch_size or batch_size
		for j = 1, batches do
			batch_lengths[j] = batch_lengths[j] +
				data["lengths"][perm[i + j - 1]]
		end
	end

	-- We need a separate array of actions to figure when each component of
	-- the network in batch mode needs to save or clear its context.
	local batch_actions = {}
	for i = 1, batch_size do
		batch_actions[i] = {}
	end

	local max_seq_len = batch_lengths:max()

	-- Used to keep track of the offsets into the `batch_data` array.
	local batch_off = torch.IntTensor(batch_size):fill(1)
	local batch_data = torch.IntTensor(batch_size, max_seq_len)

	-- Used to keep track of the length of the documents seen by each of the
	-- components of the network in batch mode. Suppose we have 10
	-- documents and the batch size is 4. Then the first component will see
	-- documents 1, 5, 9; the second, 2, 6, 10; the third, 3 and 7; and the
	-- fourth, 4 and 8. The matrix of document lengths must have three
	-- columns.
	local batch_lengths = torch.IntTensor(batch_size,
		1 + math.floor((doc_count - 1) / batch_size))

	-- Copies data from the original arrays to the arrays that are designed
	-- for batch mode processing by the network.
	local function process_document(batch_index, doc_index)
		local off = batch_off[batch_index]
		local dst_index = 1 + math.floor((doc_index - 1) / batch_size)

		local src_index = perm[doc_index]
		local src_len = data["lengths"][src_index]

		batch_data[{batch_index, {off, off + src_len - 1}}] =
			data["documents"][{src_index, {1, src_len}}]
		batch_lengths[batch_index][dst_index] = src_len

		for i = max_bptt_len, src_len - 1, max_bptt_len do
			batch_actions[batch_index][off + i] = actions["truncate_bptt"]
		end
		batch_actions[batch_index][off + src_len - 1] = actions["end_document"]

		-- Observe that we are setting this to the position of the
		-- character *after* the end of the document.
		batch_off[batch_index] = batch_off[batch_index] + src_len
	end

	for i = 1, doc_count, batch_size do
		local batches = i + batch_size - 1 > doc_count and
			doc_count % batch_size or batch_size
		for j = 1, batches do
			process_document(j, i + j - 1)
		end
	end

	return {
		data = batch_data,
		lengths = batch_lengths,
		actions = batch_actions
	}
end
