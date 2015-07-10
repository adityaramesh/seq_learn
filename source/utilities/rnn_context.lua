require "torch"

RNNContext = {}
RNNContext.__index = RNNContext

function RNNContext.create(model, vocab_size, batch_size, length)
	assert(vocab_size > 0)
	assert(batch_size > 0)
	assert(length > 0)

	local context = {
		model = model,
		vocab_size = vocab_size,
		batch_size = batch_size,
		-- The length of the RNN context in timesteps.
		length = length,

		zero_state = torch.zeros(batch_size, model.lstm_cell_width):cuda(),
		zero_prediction = torch.zeros(batch_size, vocab_size):cuda(),

		-- The states and outputs of the units in each layer.
		layer_states = {},
		-- The gradients with respect to the states and outputs of the
		-- units in each layer.
		layer_gradients = {},
		predictions = {},
		-- The predictions of the network are obtained using a dummy
		-- output module. To ensure that its gradient contributions
		-- during bprop are zero, we set the module's output gradients
		-- to zero.
		prediction_gradients = torch.zeros(batch_size, vocab_size):cuda()
	}
	
	for i = 1, length do
		context.layer_states[i] = {}
		for j = 1, 2 * model.layers do
			context.layer_states[i][j] = context.zero_state
		end
	end

	for i = 1, 2 * model.layers do
		-- The gradients with respect to the states and outputs of each
		-- layer are stored adjacently.
		layer_gradients[i] = context.zero_state
	end
	return context
end

function RNNContext:clear_state()
	for i = 0, #self.layer_states do
		self.layer_states[i] = {}
		for j = 1, #self.layer_states[i] do
			self.layer_states[i][j]:zero()
		end
	end
end

function RNNContext:clear_layer_gradients()
	for i = 1, #self.layer_gradients do
		self.layer_gradients[i]:zero()
	end
end
