require 'xlua'
require 'optim'
require 'nn'
dofile 'PartialConnected.lua'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
]] -- table

local model = nn.Sequential()

model:add(nn.PartialConnected(1,1))

-- inputs = torch.Tensor(1,8,1,1):uniform(-2,2)
model:float()
-- inputs:float()
targets = torch.Tensor(4)
for i = 1,4 do
    targets[i] = i
end

targets:float()

confusion = optim.ConfusionMatrix(4)

parameters,gradParameters = model:getParameters()
criterion = nn.CrossEntropyCriterion()
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

model:training()
for i = 1,30 do
inpu = torch.Tensor(30,8,1,1):uniform(-5,5):float()
inputs = torch.Tensor(1,8,1,1):uniform(-2,2)
inputs:float()
    local feval = function(x)
      --[[
         sgd call faval function(loss function), 
         cal the grandient wrt to parameters by plugging in parameters
         
         in sgd:
            fx,dfdx = opfunc(x) -- then use dfdx to updata the paramters
         
         ie. f/fx: current cose; gradPatameters/dfdx: grandient of loss function wrt to paramters
            f, gradParameters = feval(paramters)
      -- return parameters(has been changed) and {fx}
      --]]
      if x ~= parameters then parameters:copy(x) end
      
      gradParameters:zero()
      
      model = model:float()
     -- print(type(inputs))
      local outputs = model:forward(inpu[{{i},{}}])
      print("output is ")
  --    local outputs = model:forward(torch.FloatTensor{inputs})
      print(outputs)   
      -- self.model[i].output:addmm(0, 1, self.model[i].weight, self.model[i].input)
      -- cast datatypr
      targets = targets:float()
      outputs = outputs:float()
      criterion = criterion:float()
      
      -- calculate error/cost(f) and gradient wrt to parameters
      -- \partial{L} / \partial{W} = -2 * E = 2 * (outputs - targets)
      local f = criterion:forward(outputs, targets)
	print('criterion forward done')
      local df_do = criterion:backward(outputs, targets)
      
      -- apply chain rule, get gradPatameters for each layer
      -- in backward procedure: pass gradOutput of current layer to previous layer
      -- 
	print('backward')
      model:backward(inputs, df_do)
      
      -- add prediction to cunfusion matrix to calculate the train error
      confusion:add(outputs, targets)

        parameters,gradParameters = model:getParameters()
	-- model:parameters()
	print('paras: ',parameters)
	print('gradparameters: ')
	print(gradParameters)
      -- return cost and gradient for parameters updating
      return f,gradParameters
    end

   optim.sgd(feval, parameters, optimState) -- return paramters and {fx}

end


