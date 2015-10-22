-- word LSTM
-- packaging on module.LSTM
--[[
--]]
require 'nn'
require 'nngraph'
local LSTM = require 'module.LSTM'

local word_LSTM = {}

function word_LSTM.create(input_size, output_size, rnn_size, num_layers, dropout)
  dropout = dropout or 0
  -- word_LSTM inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  -- module.LSTM inputs
  local inputs_LSTM = {}
  for L = 1,#inputs do
    table.insert(inputs_LSTM, inputs[L])
  end
  -- module.LSTM
  local module_LSTM = LSTM.create(input_size, rnn_size, num_layers, dropout)
  local outputs_LSTM = module_LSTM(inputs_LSTM)
  -- LSTM output
  local top_h = nn.SelectTable(-1)(outputs_LSTM)
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, output_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  -- module.LSTM outputs
  local outputs = {}
  for L = 1,num_layers do
    table.insert(outputs, nn.SelectTable(L*2-1)(outputs_LSTM))
    table.insert(outputs, nn.SelectTable(L*2)(outputs_LSTM))
  end
  table.insert(outputs, logsoft)
  -- new LSTM model
  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule(inputs, outputs)
end

return word_LSTM