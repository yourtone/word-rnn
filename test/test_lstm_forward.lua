--[[
test for model.LSTM and module.LSTM
]]--

require 'nn'
require 'nngraph'
local modelLSTM = require 'model.LSTM'
local moduleLSTM = require 'module.LSTM'

local vocab_size = 300
local input_size = 30
local wordveclen = input_size
local rnn_size = 20
local output_size = 15
local num_layers = 2
local dropout = 0.5

-- inputs
local inputs = {}
table.insert(inputs, nn.Identity()()) -- x
for L = 1,num_layers do
  table.insert(inputs, nn.Identity()()) -- prev_c[L]
  table.insert(inputs, nn.Identity()()) -- prev_h[L]
end
-- inputs for module.LSTM
local x = nn.LookupTable(vocab_size, wordveclen)(inputs[1])
local inputs_origin = {}
table.insert(inputs_origin, x) -- x
for L = 2,#inputs do
  table.insert(inputs_origin, inputs[L])
end
-- module.LSTM
local rnnmodule_origin = moduleLSTM.create(input_size, rnn_size, num_layers, dropout)
local outputs = rnnmodule_origin(inputs_origin)
-- LSTM output
local top_h = nn.SelectTable(-1)(outputs)
if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
local proj = nn.Linear(rnn_size, output_size)(top_h)
local logsoft = nn.LogSoftMax()(proj)
-- outputs
local outputs_new = {}
for L = 1,num_layers do
  table.insert(outputs_new, nn.SelectTable(L*2-1)(outputs))
  table.insert(outputs_new, nn.SelectTable(L*2)(outputs))
end
table.insert(outputs_new, logsoft)
-- new LSTM model
local rnnmodule = nn.gModule(inputs, outputs_new)



local rnnmodel = modelLSTM.lstm(vocab_size, wordveclen, output_size, rnn_size, num_layers, dropout)

print('Start test ...')
local ins1 = {torch.Tensor{3}, torch.rand(1, rnn_size), torch.rand(1, rnn_size), 
              torch.rand(1, rnn_size), torch.rand(1, rnn_size)}
local outs1 = rnnmodel:forward(ins1)
print('Output model:')

print(outs1)
local ins2 = {torch.Tensor{3}, torch.rand(1, rnn_size), torch.rand(1, rnn_size), 
              torch.rand(1, rnn_size), torch.rand(1, rnn_size)}
local outs2 = rnnmodule:forward(ins2)
print('Output module:')
print(outs2)
