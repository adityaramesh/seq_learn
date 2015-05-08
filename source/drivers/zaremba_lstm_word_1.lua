require "source/utilities/zaremba/lstm"

params = {
	-- Optimization.
	bptt_len      = 100,
	batch_size    = 20,
	init_lr       = 1,
	lr_decay      = 2,
	lr_decay_rate = 4,
	max_grad_norm = 5,
	dropout_prob  = 0,

	-- Architecture.
	vocab_size          = 10000,
	layers              = 1,
	lstm_cell_width     = 400,
	max_init_weight_mag = 0.1,
	model_granularity   = "word"
}
run(params)
