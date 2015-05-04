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

	for corpus in data[category] do
		if corpus["contents"] == nil then
			error("\"contents\" key must be present for all corpora.")
		end

		if corpus["lengths"] ~= nil then
			-- The corpus consists of multiple documents.
			max_doc_len = math.max(max_doc_len, corpus["contents"]:size(2))
			doc_count = doc_count + corpus["contents"]:size(1)
		else
			-- The corpus consists of only one document.
			max_doc_len = math.max(max_doc_len, corpus["contents"]:size(1))
			doc_count = doc_count + 1
		end
	end

	-- Now we allocate the memory and perform the actual copying.
	docs = torch.IntTensor(doc_count, max_doc_len):zero()
	lengths = torch.IntTensor(doc_count)
	cur_doc = 1

	for corpus in data[category] do
		if corpus["lengths"] ~= nil then
			rows = corpus["contents"]:size(1)
			cols = corpus["contents"]:size(2)
			docs[{{cur_doc, cur_doc + rows}, {1, cols}}] = corpus["contents"]
			lengths[{{cur_doc, cur_doc + rows}}] = corpus["lengths"]
			cur_doc = cur_doc + rows + 1
		else
			docs[cur_doc] = corpus["contents"]
			cur_doc = cur_doc + 1
		end
	end

	return cur_doc, lengths
end

function load_hdf5(fn)
	local data = hdf5.open(fn):read():all()
	local train_docs, train_lengths = merge_documents("train", data)
	local validate_docs, validate_lengths = merge_documents("validate", data)

	new_data = {
		vocab = data["vocab"],
		train = {
			documents = train_docs,
			lengths = train_lengths
		},
		validate = {
			documents = validate_docs,
			lengths = validate_lengths
		}
	}

	data = {}
	collectgarbage()
	return new_data
end

function shuffle_documents(batch_size, max_bptt_len)

end
