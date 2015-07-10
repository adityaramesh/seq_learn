require "source/utilities/lstm_rnn"

data = torch.FloatTensor({
	1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
	1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
	1, 1, 0, 1, 1, 0, 1, 1, 1, 0
}):reshape(4, 10):cuda()

token_ends_doc = torch.IntTensor({
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
	0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
	0, 0, 1, 0, 0, 1, 0, 0, 0, 1
}):reshape(4, 10)

params = {
	layers = 1,
	vocab_size = 50,
	dropout_prob = 0,
	lstm_cell_width = 400,

	length = 10,
	batch_size = 4,
	max_init_weight_mag = 0.1
}

net = LSTM_RNN.create(params)
for i = 1, 2 do
	-- TODO: make sure that we are using token_ends_doc correctly: are we
	-- summing along the batch_size or the window width?
	-- The error appears to be using data[{{}, i}] for i > 1
	net:forward(i, data[{{}, i}], data[{{}, i + 1}], token_ends_doc[{{}, i}])
end
-- TODO test forward, backward, advance
