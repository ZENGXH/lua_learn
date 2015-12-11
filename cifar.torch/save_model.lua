require 'torch'
require 'nn'
function saveModel(model, filename)
--require 'torch'
--require 'nn'
local emptyModel = model:clone('weight', 'bias')

torch.save(filename,emptyModel)
print('current model is saved as')
print(filename)

end
