require "torch"
require "cunn"
require "nngraph"
local LookupTable = nn.LookupTable

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
		batch_size = params.batch_size,
		network = module,
		criterion = nn.ClassNLLCriterion():cuda(),

		-- Used to keep track of which tokens determine the end of
		-- documents in the current window. Important for figuring out
		-- when to use a new state instead of the previous one, or when
		-- to use a zero input gradient instead of the next one.
		token_ends_doc = torch.ByteTensor(params.length, params.batch_size):zero(),
		last_doc_endings = torch.IntTensor(params.batch_size):zero(),

		-- A useful constant of which we will make many shallow copies.
		zero_layer_state = torch.zeros(params.batch_size,
			params.lstm_cell_width):cuda(),

		-- The predictions of the network are obtained using a dummy
		-- output module. To ensure that its gradient contributions
		-- during bprop are zero, we set the module's output gradients
		-- to zero.
		pred_grads = torch.zeros(params.batch_size,
			params.vocab_size):cuda()

		preds = {},
		states = {},

		-- The states supplied to the first clone during the forward
		-- pass.
		zero_state = {},
		-- Used when the current token along any of the `batch_size`
		-- components of the input is the beginning of a document. Then
		-- the next state will be a mixture of zeros and previous
		-- states.
		prev_states = {},
		-- The state gradients supplied to the last clone during the
		-- backward pass.
		zero_state_grads = {},
		-- Used when the current token along any of the `batch_size`
		-- componenst of the input is the end of a document. Then the
		-- state gradients will be a mixture of zeros and state
		-- gradients from the next clone.
		state_grad_buf = {},
		next_state_grads = {}
	}

	for i = 1, 2 * model.layers do
		model.zero_state[i] = model.zero_state
	end

	for i = 2, model.length do
		model.prev_states[i] = {}
		for j = 1, 2 * model.layers do
			model.prev_states[i][j] = model.zero_state:clone()
		end
	end

	for i = 1, 2 * model.layers do
		model.zero_state_grads[i] = model.zero_state
		model.state_grad_buf[i] = model.zero_state:clone()
	end
	return model
end

--
-- This function must be called after the network is created or deserialized.
--
function LSTM_RNN:restore()
	self.clones = clone_network(self.module, self.length)
end

function LSTM_RNN:forward(i, input, output, token_ends_doc)
	assert(i >= 1 and i <= self.length)
	assert(input:size(1) == params.batch_size)
	assert(output:size(1) == params.batch_size)
	assert(token_ends_doc:size(1) == params.batch_size)

	self.token_ends_doc[i]:copy(token_ends_doc)
	if i == 1 then
		self.last_doc_endings:zero()
	end
	for j = 1, token_ends_doc:size(1) do
		if token_ends_doc[j] == 1 then
			self.last_doc_endings[j] = i
		end
	end

	local prev_state
	if i == 1 then
		prev_state = self.zero_state
	else if self.token_ends_doc[i - 1]:sum() == 0 then
		prev_state = self.states[i - 1]
	else
		prev_state = self.prev_state[i]
		for j = 1, 2 * model:layers() do
			prev_state[j]:copy(self.states[i - 1][j])
		end
		for j = 1, self.token_ends_doc:size(2) do
			if self.token_ends_doc[i - 1][j] == 1 then
				for k = 1, 2 * model:layers() do
					prev_state[k][j]:zero()
				end
			end
		end
	end

	self.states[i], self.preds[i] = unpack(self.clones[i]:forward(
		{input, prev_state}))

	if output == nil then return end
	return self.criterion:forward(self.preds[i], output)
end

function LSTM_RNN:backward(i, input, output)
	assert(i >= 1 and i <= self.length)
	assert(input:size(1) == params.batch_size)
	assert(output:size(1) == params.batch_size)

	local output_grads = self.criterion:backward(self.preds[i], output)
	for j = 1, self.last_doc_endings:size(1) do
		if i > self.last_doc_endings[j] then
			output_grads[j]:zero()
		end
	end

	local prev_state
	if i == 1 then
		prev_state = self.zero_state
	else if self.token_ends_doc[i - 1]:sum() == 0 then
		prev_state = self.states[i - 1]
	else
		prev_state = self.prev_state[i]
	end

	local next_state_grads
	if i == self.batch_size then
		next_state_grads = self.zero_state_grads
	else if self.token_ends_doc[i]:sum() == 0 then
		next_state_grads = self.next_state_grads
	else
		next_state_grads = self.state_grad_buf
		for j = 1, 2 * model:layers() do
			next_state_grads[j]:copy(self.next_state_grads[j])
		end
		for j = 1, self.token_ends_doc:size(2) do
			if self.token_ends_doc[i][j] == 1 then
				for k = 1, 2 * model:layers() do
					next_state_grads[k][j]:zero()
				end
			end
		end
	end

	self.state_grads = self.clones[i]:backward({input, prev_state},
		{output_grads, next_state_grads, self.pred_grads})[2]
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
