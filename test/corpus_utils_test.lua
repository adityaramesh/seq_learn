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
max_bptt_len = 200
batch_data = batch_documents(batch_size, max_bptt_len, data["train"])

lines = {}
for i = 1, batch_size do
	cur_line = {""}
	for j = 1, batch_data["data"]:size(2) do
		cur_line[#cur_line + 1] = data["vocab"][batch_data["data"][i][j]]
		if batch_data["actions"][i][j] == actions["end_document"] then
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
