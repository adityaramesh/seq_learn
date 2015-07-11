dofile "source/utilities/corpus_utils.lua"

orig_data_fn = "data/ptb/ptb.train.txt"
data = load_hdf5("data/preprocessed/ptb.hdf5")

-- Compare the data read back from the file to the original.
print("Checking consistency of serialized data.")
lines = {}
cur_line = {""}

for i = 1, data["train"]["documents"]:size(1) do
	for j = 1, data["train"]["lengths"][i] do
		cur_line[#cur_line + 1] = data["vocab"][data["train"]["documents"][i][j]]
	end
	cur_line[#cur_line + 1] = ""
	lines[#lines + 1] = table.concat(cur_line, " ")
	cur_line = {""}
end

lines_set = {}
for index, line in pairs(lines) do
	lines_set[line] = 0
end

-- Check that every line in the original data set is found in the condensed data
-- set.
fh = io.open(orig_data_fn)
while true do
	line = fh:read()
	if line == nil then break end
	line = line:gsub("\n", "")
	if lines_set[line] == nil then
		print("Line not found: \"" .. line .. "\"")
		break
	end
end

-- Compare the batch-processed data to the original.
print("Checking consistency of batch-processed data.")
batch_size = 20
batch_data = batch_documents(batch_size, data["train"])

lines = {}
for i = 1, batch_size do
	cur_line = {""}
	cur_doc_index = 1
	for j = 1, batch_data["lengths"][i] do
		cur_line[#cur_line + 1] = data["vocab"][batch_data["data"][i][j]]
		if j == batch_data["boundaries"][i][cur_doc_index] then
			cur_doc_index = cur_doc_index + 1
			cur_line[#cur_line + 1] = ""
			lines[#lines + 1] = table.concat(cur_line, " ")
			cur_line = {""}
		end
	end
end

lines_set = {}
for index, line in pairs(lines) do
	lines_set[line] = 0
end

fh = io.open(orig_data_fn)
while true do
	line = fh:read()
	if line == nil then break end
	line = line:gsub("\n", "")
	if lines_set[line] == nil then
		print("Line not found: \"" .. line .. "\"")
		break
	end
end
