require 'nn'
require 'torch'
local PartialConnected, parent = torch.class('nn.PartialConnected', 'nn.Module')


function PartialConnected: __init(inputSize, outputSize)
	parent.__init(self)

	inputSize = 8
	outputsize = 4
	self.model = {}
	input_split_flag = 0
	gradOutput_split_flag = 0

--[[
	function createModel(inputSize, outputSize)
		-- inputSize and be {N, 2}
		submodel = {}
		submodel.input = torch.Tensor(inputSize)
		submodel.weight = torch.Tensor(inputSize, outputSize):uniform(-1,1)
		submodel.bias = torch.Tensor(outputSize)
		submodel.gradWeight = torch.Tensor(inputSize, outputSize)
		submodel.gradBias = torch.Tensor(outputSize)
		submodel.output = torch.Tensor(outputSize)
		submodel.gradOutput = torch.Tensor(outputSize)
		print(submodel)		
		return submodel

	end
--]]
        num_model = 4
	self.model = {}
	for i=1, num_model do
		table.insert(self.model, createModel({2,2}, {2,1}))
        end	

	print(self.model)
	-- #
	-- three in a mask
--	parent.__init(self)
	self.weight = torch.Tensor(1)
	self:updateModel2self()
	
	self:reset()
end


        function createModel(inputSize, outputSize)
                -- inputSize and be {N, 2}
                submodel = {}
                submodel.input = torch.Tensor(inputSize)
                submodel.weight = torch.Tensor(2, 1):uniform(-1,1)
                submodel.bias = torch.Tensor(outputSize)
                submodel.gradWeight = torch.Tensor(2, 1)
                submodel.gradBias = torch.Tensor(outputSize)
                submodel.output = torch.Tensor(outputSize)
                submodel.gradOutput = torch.Tensor(outputSize)
		print('submodelweight')
                print(submodel.weight:size())
                return submodel
        end



function PartialConnected:updateModel2self()
	self.weight = self.model[1].weight
	self.bias = self.model[1].bias
	self.gradWeight = self.model[1].gradWeight
	self.gradBias = self.model[1].gradBias

	for i = 2, #self.model do
		self.weight = torch.cat(self.weight, self.model[i].weight)
		self.bias = torch.cat(self.bias, self.model[i].bias)
		self.gradWeight = torch.cat(self.gradWeight, self.model[i].gradWeight)
		self.gradBias = torch.cat(self.gradBias, self.model[i].gradBias)
	end
	print('self parameters update')
end


function PartialConnected:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.model[1].weight:size()[1])
   end

   for i = 1, #self.model do
      self.model[i].weight:uniform(-stdv, stdv)
      self.model[i].bias:uniform(-stdv, stdv)
   end

   return self
end




function PartialConnected:updateOutput(input) 
	-- input size； BATCH * D
	-- input dimension should be 12 x 1 vector
	-- output = torch.Tensor(4):fill(0) -- initilaize
	if(input_split_flag) then self:splitInput2Sub(input) end

	if(input:dim() ~= 1) then 
		-- reshape()
                self.addBuffer = self.addBuffer or self.model[1].input.new()
                self.addBuffer:fill(1)
		for i = 1, #self.model do -- 1 to 4
			self.model[i].input:resize(2,2)
			print(self.model[i].output:size())
			print(self.model[i].bias:size())
			print(self.model[i].weight:size())
			print(self.model[i].input:size())
			-- self.model[i].input:reshape(2,1)i
			M = torch.Tensor(2, 1)
			torch.addmm(M:float(), self.model[i].input:float(),  self.model[i].weight)
			print(M)
			--self.model[i].output:addmm(0, self.model[i].output, 1, self.model[i].input,
			--		      self.model[i].weight:t())
			self.model[i].output:addmm(self.model[i].input,
					      self.model[i].weight:t())
			
		--	self.model[i].output = torch.add(self.model[i].input * self.model[i].weight + self.model[i].bias
			
			self.model[i].output:addr(1, self.addBuffer, self.model[i].bias)
			-- use torch.addmv([res,] [beta,] [v1,] vec1, [v2,] mat, vec2)
			-- torch.addmv(1Db_vec(N,D), X(N,D), W(D))
			-- res = (beta * res) + (v1 * vec1) + (v2 * (mat * vec2))
			-- beta: momentumn, v1: bias, v2 = 1, mat: X, vec2: weight
			print(self.model[i].output)
		end
                -- self.addBuffer = self.addBuffer or self.model[1].input.new()
                -- self.addBuffer:fill(1)
	
	else
		print('dimension error')
		-- sli = input:reshape(input:size(1),4,2) 
		-- for i = 1, #sli[1] do -- 1 to 4
		-- output[i] = bias[i] + weight[i] * sli[i] -- use torch.addmv
		-- output:addmv()
	end

	-- cat output
	local out = self.model[1].output:clone()

	for i = 2, #self.model do
		out = torch.cat(out, self.model[i].output)
	end

	self.output = out

	return self.output -- 4x1
end
--[[
function PatialConnected:backward(intput, gradOutput)
	bp, gradOutput
		(pass by the back layer 
		base to the output of current layer and target) 

		return by 
			`MSECriterion_updateOutput`
				use new output 
				run computeCost and gradient again then call
			`MSECriterion_updateGradInput` 
				=> store the result of last computation of 
				the gradients of the loss function(by criterion.backward)
	   
	   function task: call 
			`updateGradInput(input, gradOutput)` 
			`accGradParameters(input,gradOutput,scale)`
				if only one layer, gradOutput is exactly: 
					gradOutput is gradients compute from criterion
					= \partial{(y - W^T X)} / \partial{W} 
					= 1/N X'* (y - W^T X)
					= 1/N X'* (target - outputResult), 
					X is the input of the final layer
				if more then one layer, by BP,
					gradient wrt to parameters
					= gradOutput 
	 
end --]]

function PartialConnected:accGradParameters(input, gradOutput, scale)
	--[[ 
		`GradParameter` means grad wrt to weight and grad wrt bias

		`scale` is a scale factor 
			that is multiplied with the gradParameters before being accumulated.
			(not momentumn! here is calculating current gradient, not the g for parameter updating)

		`gradient wrt to W` = gradWeight 
			= GET * \patial{output}/\partial{para} 
			= GET * X 
			= input * gradOutput	 
		
		gradient wrt to bias:
			= GET * \partial{output = W^T + b}/\partial{b}
			= GET
		bias = step * GET
	--]]

	scale = scale or 1
	if(not input_split_flag) then splitInput2Sub(input) end
	if(not gradOutput_split_flag) then splitGradOutput2Sub(gradOutput) end

	if(input:dim() ~= 1) then 

		--[[
			[res] torch.addr([res,] [v1,] mat, [v2,] vec1, vec2)
				= mat(M,N) + vec1(M,1)vec1'(1,N)
			notice that, if write as: 
				a(D,1):addr(scale, gradOutput(D, 1), input(D, 1))
				=> a = scale * a + gradOutput(D, 1) input'(1, D)

			ie, gradWeight(12 = D1(3) + D1(3) + D3(3) + D4(3), 1) 
				= (gradWeight * scale -- ) + gradOutput(D, 1) x input
		--]]

		-- check dimention: reshape() first dim is #model
		-- sli = input:reshape(4, 2) -- (#model, #ft)
		-- self.gradWeight:reshape(4,3) -- 
		-- gradOutput: reshape(4, 1)

		for i = 1, #self.model do
			-- ?? self.weight[i].addr(scale, gradOutput[i], sli[i])

			self.model[i].gradWeight:addmm(scale, 
										self.model[i].gradOutput:t(),
										self.model[i].input) 
			self.model[i].gradBias:addmv(scale,
										self.model[i].gradOutput:t(),
										self.model[i].addBuffer) 
	         end  		-- in the case of output is scala
        else 
		print('input dimension should be one')
	end

	self:updateModel2self()

end

function PartialConnected:updateGradInput(input, gradOutput)
	--[[ 
		compute gradient of the module wrt its own input
		return: gradInput, 4 model, each in dimension 2 + 1, 
		4 X 3 dimension 

		gradOutput in shape(#current layer output, 1)
		
		pass to the previous layer: 
		PASS 
			= GET * \partial{output}/\partial{input}
			= GET * W
	--]]
	if(not input_split_flag) then splitInput2Sub(input) end

	if(not gradOutput_split_flag) then splitGradOutput2Sub(gradOutput) end

	if input:dim() ~= 1 then
		for i = 1,#self.model do
			self.model[i].gradInput:addmv(0, 1, self.model[i].gradOutput, 
						     self.model[i].weight)
		end
	end

	self:updateModel2self()
end


function PartialConnected:splitInput2Sub(input)
	input_split_flag = 1
	for i=1, #self.model do
		self.model[i].input = input[{{}, 
			{1 + (i-1) * 2, i * 2}}]:float()
		print('split')
		print(self.model[i].input:size())
        end
end

function splitGradOutput2Sub(gradOutput)
	gradOutput_split_flag = 1
	for i=1, #self.model do
		self.model[i].gradOutput = gradOutput[{{}, 
			{1 + (i-1) * #self.model[i].output, i * #self.model[i].output}}]
		self.model[i].gradOutput = gradOutput:reshape(self.model[i].gradOutput:size()[1],1)
        end
end
