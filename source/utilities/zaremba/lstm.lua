--
-- This file is a modified version of Zaremba's code, with added support for
-- `serialization.lua`. The purpose of this code is to allow us to compare our
-- results to those of the models from his paper more easily.
--
-- Example parameters object to be supplied to `start()`:
-- params = {
-- 	-- Optimization.
-- 	bptt_len      = 100,
-- 	batch_size    = 20,
-- 	init_lr       = 1,
-- 	lr_decay      = 2,
-- 	lr_decay_rate = 4,
-- 	max_grad_norm = 5,
-- 	dropout_prob  = 0,
--
-- 	-- Architecture.
-- 	vocab_size          = 50,
-- 	layers              = 1,
-- 	lstm_cell_width     = 400,
-- 	max_init_weight_mag = 0.1,
-- 	model_granularity   = "character"
-- }
--

require "cunn"
require "nngraph"
require "xlua"
local LookupTable = nn.LookupTable

require "source/utilities/serialization"
require "source/utilities/zaremba/base"
local ptb = require "source/utilities/zaremba/data"

local function transfer_data(x)
	return x:cuda()
end

local function perplexity(mean_nll, params)
	if params.model_granularity == "word" then
		return torch.exp(mean_nll)
	else
		return torch.exp(5.6 * mean_nll)
	end
end

--
-- Creates an LSTM unit. In Zaremba's LSTM RNN architecture, there is only one
-- LSTM unit per layer. The widths of the cells within each unit are determined
-- by `params.lstm_cell_width`.
--
-- Parameters:
-- - `input`: The input to the current layer.
-- - `prev_cell_state`: Cell state of the same LSTM unit from timestep `t - 1`.
-- - `prev_output`: Output of the same LSTM unit from timestep `t - 1`.
--
local function make_lstm_unit(input, prev_cell_state, prev_output, params)
	-- Creates the modules that compute the activations for each of the four
	-- gates.
	local function make_gate_act_module()
		local input_contrib = nn.Linear(params.lstm_cell_width,
			params.lstm_cell_width)
		local prev_output_contrib = nn.Linear(params.lstm_cell_width,
			params.lstm_cell_width)
		return nn.CAddTable()({input_contrib(input),
			prev_output_contrib(prev_output)})
	end

	local input_gate     = nn.Sigmoid()(make_gate_act_module())
	local forget_gate    = nn.Sigmoid()(make_gate_act_module())
	local input_mod_gate = nn.Tanh()(make_gate_act_module())
	local cur_cell_state = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_cell_state}),
		nn.CMulTable()({input_gate, input_mod_gate})
	})

	local output_gate     = nn.Sigmoid()(make_gate_act_module())
	local cur_cell_output = nn.CMulTable()({output_gate, nn.Tanh()(cur_cell_state)})
	return cur_cell_state, cur_cell_output
end

local function make_network(params)
	local x      = nn.Identity()()
	local y      = nn.Identity()()
	local prev_s = nn.Identity()()
	local i      = {[0] = LookupTable(params.vocab_size, params.lstm_cell_width)(x)}
	local next_s = {}
	local split  = {prev_s:split(2 * params.layers)}
	for layer_idx = 1, params.layers do
		local prev_c         = split[2 * layer_idx - 1]
		local prev_h         = split[2 * layer_idx]
		local dropped        = nn.Dropout(params.dropout_prob)(i[layer_idx - 1])
		local next_c, next_h = make_lstm_unit(dropped, prev_c, prev_h, params)
		table.insert(next_s, next_c)
		table.insert(next_s, next_h)
		i[layer_idx] = next_h
	end
	local h2y     = nn.Linear(params.lstm_cell_width, params.vocab_size)
	local dropped = nn.Dropout(params.dropout_prob)(i[params.layers])
	local pred    = nn.LogSoftMax()(h2y(dropped))
	local err     = nn.ClassNLLCriterion()({pred, y})

	local module = nn.gModule({x, y, prev_s}, {err, nn.Identity()(next_s),
		nn.Identity()(pred)})
	module:getParameters():uniform(-params.max_init_weight_mag,
		params.max_init_weight_mag)
	return transfer_data(module)
end

local function get_model_info(params)
	local model_info = {}
	model_info.s = {}
	model_info.ds = {}
	model_info.start_s = {}
	model_info.dummy_output_grads = transfer_data(torch.zeros(params.batch_size,
		params.vocab_size))

	for j = 0, params.bptt_len do
		model_info.s[j] = {}
		for d = 1, 2 * params.layers do
			model_info.s[j][d] = transfer_data(torch.zeros(
				params.batch_size, params.lstm_cell_width))
		end
	end
	for d = 1, 2 * params.layers do
		model_info.start_s[d] = transfer_data(
			torch.zeros(params.batch_size, params.lstm_cell_width))
		model_info.ds[d] = transfer_data(
			torch.zeros(params.batch_size, params.lstm_cell_width))
	end

	model_info.network = make_network(params)
	model_info.norm_dw = 0
	model_info.err = transfer_data(torch.zeros(params.bptt_len))
	return model_info
end

local function initialize_context(params, info)
	local context = {}
	context.paramx, context.paramdx = info.model.network:getParameters()
	context.rnns = g_cloneManyTimes(info.model.network, params.bptt_len)
	return context
end

local function get_train_info(params)
	local train_info = {}
	train_info.iter          = 0
	train_info.epoch         = 1
	train_info.bptt_len      = params.bptt_len
	train_info.batch_size    = params.batch_size
	train_info.learning_rate = params.init_lr
	train_info.lr_decay      = params.lr_decay
	train_info.lr_decay_rate = params.lr_decay_rate
	train_info.max_grad_norm = params.max_grad_norm
	train_info.dropout_prob  = params.dropout_prob

	train_info.train = {pos = 1}
	train_info.valid = {pos = 1}
	return train_info
end

local function reset_state(data, mode, params, info)
	info.train[mode].pos = 1
	if info.model ~= nil and info.model.start_s ~= nil then
		for d = 1, 2 * params.layers do
			info.model.start_s[d]:zero()
		end
	end
end

local function reset_ds(info)
	for d = 1, #info.model.ds do
		info.model.ds[d]:zero()
	end
end

local function clear_model_context(params, info)
	local dummy_data = {train = {}}
	reset_state(dummy_data, "train", params, info)
	reset_ds(info)

	for j = 0, info.train.bptt_len do
		for d = 1, 2 * params.layers do
			info.model.s[j][d]:zero()
		end
	end
end

local function forward(data, mode, params, info, context)
	g_replace_table(info.model.s[0], info.model.start_s)
	if info.train[mode].pos + info.train.bptt_len > data.data:size(1) then
		reset_state(data, mode, params, info)
	end

	for i = 1, info.train.bptt_len do
		local x = data.data[info.train[mode].pos]
		local y = data.data[info.train[mode].pos + 1]
		local s = info.model.s[i - 1]

		info.model.err[i], info.model.s[i] = unpack(
			context.rnns[i]:forward({x, y, s}))
		info.train[mode].pos = info.train[mode].pos + 1
	end
	g_replace_table(info.model.start_s, info.model.s[info.train.bptt_len])
	return info.model.err:mean()
end

function backward(data, params, info, context)
	context.paramdx:zero()
	reset_ds(info)

	for i = info.train.bptt_len, 1, -1 do
		info.train.train.pos = info.train.train.pos - 1
		local x = data.data[info.train.train.pos]
		local y = data.data[info.train.train.pos + 1]
		local s = info.model.s[i - 1]
		local derr = torch.ones(1):cuda()
		local tmp = context.rnns[i]:backward({x, y, s}, {derr,
			info.model.ds, info.model.dummy_output_grads})[3]

		g_replace_table(info.model.ds, tmp)
		cutorch.synchronize()
	end

	info.train.train.pos = info.train.train.pos + info.train.bptt_len
	info.model.norm_dw = context.paramdx:norm()

	if info.model.norm_dw > info.train.max_grad_norm then
		local shrink_factor = info.train.max_grad_norm / info.model.norm_dw
		context.paramdx:mul(shrink_factor)
	end
	context.paramx:add(context.paramdx:mul(-info.train.learning_rate))
end

function validate(data, params, info, context)
	reset_state(data.data, "valid", params, info)
	g_disable_dropout(context.rnns)
	local len = (data.data:size(1) - 1) / (info.train.bptt_len)
	local perp = 0
	for i = 1, len do
		perp = perp + forward(data, "valid", params, info, context)
	end

	perp = perplexity(perp / len, params)
	g_enable_dropout(context.rnns)
	return perp
end

function run(params)
	ptb.init(params)
	print("Loading training data.")
	local train_data = {data = transfer_data(ptb.traindataset(params.batch_size))}
	print("Loading validation data.")
	local valid_data = {data = transfer_data(ptb.validdataset(params.batch_size))}

	local do_train, do_test, paths, info = restore(
		function() return get_model_info(params) end,
		function() return get_train_info(params) end
	)
	local context = initialize_context(params, info)

	local total_cases = 0
	local start_time = torch.tic()
	local inputs_per_epoch = torch.floor(train_data.data:size(1) / info.train.bptt_len)
	print("\nStarting epoch " .. info.train.epoch .. ".")

	while true do
		local perp = forward(train_data, "train", params, info, context)
		if info.train.perps == nil then
			info.train.perps = torch.FloatTensor(inputs_per_epoch):fill(perp)
		end
		info.train.perps[info.train.iter % inputs_per_epoch + 1] = perp
		backward(train_data, params, info, context)

		total_cases = total_cases + info.train.bptt_len * info.train.batch_size
		xlua.progress(info.train.iter % inputs_per_epoch + 1, inputs_per_epoch)
		info.train.iter = info.train.iter + 1

		if info.train.iter % torch.round(inputs_per_epoch / 10) == 10 then
			local wps = torch.floor(total_cases / torch.toc(start_time))
			local since_beg = g_d(torch.toc(start_time) / 60)
			perp = perplexity(info.train.perps:mean(), params)

			local new_best = false
			if perp < info.acc.best_train then
				info.acc.best_train = perp
				new_best = true
			end
			save_train_progress(new_best, paths, info)

			print('epoch = '                  .. g_f3(info.train.epoch)   ..
				', train perp. = '        .. g_f3(perp)               ..
				', wps = '                .. wps                      ..
				', dw:norm() = '          .. g_f3(info.model.norm_dw) ..
				', lr = '                 .. info.train.learning_rate ..
				', since last restart = ' .. since_beg                .. ' mins.')
		end

		if info.train.iter % inputs_per_epoch == 0 then
			perp = validate(valid_data, params, info, context)
			print("Validation set perplexity: " .. g_f3(perp))

			local new_best = false
			if perp < info.acc.best_test then
				info.acc.best_test = perp
				new_best = true
			end
			save_test_progress(new_best, paths, info)

			info.train.epoch = info.train.epoch + 1
			if info.train.epoch > info.train.lr_decay_rate then
				info.train.lr = info.train.lr / info.train.lr_decay
			end

			xlua.progress(inputs_per_epoch, inputs_per_epoch)
			print("\nStarting epoch " .. info.train.epoch .. ".")
		end

		if info.train.iter % 33 == 0 then
			cutorch.synchronize()
			collectgarbage()
		end
	end
end
