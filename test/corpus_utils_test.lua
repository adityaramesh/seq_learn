dofile "source/utilities/corpus_utils.lua"

data = load_hdf5("data/preprocessed/ptb.hdf5")

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

fh = io.open("data/ptb/ptb.train.txt")
while true do
	line = fh:read()
	if line == nil then break end
	line = line:gsub("\n", "")
	if lines_set[line] == nil then
		print("\"" .. line .. "\"")
		break
	end
end

batch_data = batch_documents(20, 200, data["train"])
-- TODO test
