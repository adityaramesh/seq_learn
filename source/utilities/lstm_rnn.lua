require "torch"

LSTM_RNN = {}
LSTM_RNN.__index = LSTM_RNN

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
-- Adapted from Zaremba's LSTM code.
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
	local cur_lstm_output = nn.CMulTable()({output_gate, nn.Tanh()(cur_cell_state)})
	return cur_cell_state, cur_lstm_output
end

--
-- Adapted from Zaremba's LSTM code.
--
local function clone_network(network, count)
	local params, grad_params = network:parameters()
	if params == nil then
		params = {}
	end

	local params_no_grad
	if net.parametersNoGrad then
		params_no_grad = net:parametersNoGrad()
	end

	local clones = {}
	local mem = torch.MemoryFile('w'):binary()
	mem:writeObject(network)

	for i = 1, count do
		-- We need to use a new reader for each clone.
		-- We don't want to use the pointers to already read objects.
		local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
		local clone = reader:readObject()
		reader:close()

		local clone_params, clone_grad_params = clone:parameters()
		for i = 1, #params do
			clone_params[i]:set(params[i])
			clone_grad_params[i]:set(grad_params[i])
		end

		if params_no_grad then
			local clone_params_no_grad = clone:parametersNoGrad()
			for i = 1, #params_no_grad do
				clone_params_no_grad[i]:set(params_no_grad[i])
			end
		end

		clones[i] = clone
		collectgarbage()
	end

	mem:close()
	return clones
end

function LSTM_RNN.create(params)
	-- Used to access the states of the RNN at the previous timestep.
	local prev_states       = nn.Identity()()
	local prev_states_array = {prev_states:split(2 * params.layers)}

	-- Stores the states of the RNN at the current timestep. The cell states
	-- and LSTM unit outputs from each layer are stored adjacently.
	local cur_states  = {}
	local lut         = nn.LookupTable(params.vocab_size, params.lstm_cell_width)(input)
	local prev_output = lut

	for i = 1, params.layers do
		-- Get the states of the same LSTM unit from the previous
		-- timestep.
		local prev_cell_state = prev_states_array[2 * i - 1]
		local prev_lstm_output = prev_states_array[2 * i]

		local layer_input = nn.Dropout(params.dropout_prob)(prev_output)
		local cur_cell_state, cur_lstm_output = make_lstm_unit(
			layer_input, prev_cell_state, prev_lstm_output, params)

		table.insert(cur_states, cur_cell_state)
		table.insert(cur_states, cur_lstm_output)
		prev_output = cur_lstm_output
	end

	-- Construct the prediction layer.
	local pred_inputs = nn.Dropout(params.dropout_prob)(prev_output)
	local pred_func   = nn.Linear(params.lstm_cell_width, params.vocab_size)
	local pred        = nn.LogSoftMax()(pred_func(pred_inputs))

	-- Construct the module.
	local input  = nn.Identity()()
	local module = nn.gModule({input, prev_states}, {pred,
		nn.Identity(cur_states), nn.Identity()(pred)}):cuda()
	module:getParameters():uniform(-params.max_init_weight_mag,
		params.max_init_weight_max)

	local model = {
		length = params.length,
		layers = params.layers,
		network = module,
		criterion = nn.ClassNLLCriterion():cuda(),

		-- Used for the previous state vector supplied to the first
		-- clone during the forward pass, and for the state gradients
		-- supplied to the last clone during the backward pass.
		zero_state = torch.zeros(params.batch_size,
			params.lstm_cell_width):cuda(),

		-- The predictions of the network are obtained using a dummy
		-- output module. To ensure that its gradient contributions
		-- during bprop are zero, we set the module's output gradients
		-- to zero.
		pred_grads = torch.zeros(params.batch_size,
			params.vocab_size):cuda()

		preds = {},
		states = {},
		state_grads = {},
		zero_state_grads = {}
	}

	for i = 1, 2 * model.layers do
		model.zero_state_grads[i] = model.zero_state
	end
	return model
end

--
-- This function must be called after the network is created or deserialized.
--
function LSTM_RNN:restore()
	self.clones = clone_network(self.module, self.length)
end

function LSTM_RNN:forward(i, input, output)
	assert(i >= 1 and i <= self.length)
	assert(input:size(1) == output:size(1))

	local cur_state = {}
	local prev_state = i == 1 and self.zero_state or self.states[i - 1]
	self.states[i], self.preds[i] = unpack(self.clones[i]:forward({input, s}))

	if output == nil then return end
	return self.criterion:forward(self.preds[i], output)
end

--
-- The `mask` parameter is used to nullify the gradient contributions of certain
-- components of the network in batch mode.
--
function LSTM_RNN:backward(i, input, output, mask)
	assert(i >= 1 and i <= self.length)
	assert(input:size(1) == output:size(1))

	local output_grads = self.criterion:backward(self.preds[i], output)
	if mask ~= nil then
		assert(mask:size(1) == input:size(1))
		for i = 1, #mask do
			if mask[i] == 0 then
				output_grads[i]:zero()
			end
		end
	end

	local prev_state = i == 1 and self.zero_state or self.states[i - 1]
	local prev_state_grads = i == self.length and self.zero_state_grads or
		self.state_grads
	self.state_grads = self.clones[i]:backward({input, prev_state},
		{output_grads, prev_state_grads, self.pred_grads})[2]
end

--
-- Adapted from Zaremba's LSTM code.
--
local function disable_dropout(node)
	if type(node) == "table" and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(disable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = false
	end
end

--
-- Adapted from Zaremba's LSTM code.
--
local function enable_dropout(node)
	if type(node) == "table" and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(enable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = true
	end
end

function LSTM_RNN:train()
	enable_dropout(self.clones)
end

function LSTM_RNN:evaluate()
	disable_dropout(self.clones)
end
