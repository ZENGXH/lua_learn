local Linear, parent = torch.class('nn.PatialConnected', 'nn.Module')


function PatialConnected: __init(inputSize, outputSize)
	batchSize = 1
	
	inputSize = 8
	outputsize = 4
	num_model = 4
	input_pmodel = 2 --dimension of feature
	output_pmodel = 1
	self.model = {}
	input_split_flag = false
	gradOutput_split_flag = false

	for i=1, num_model do 
		table.insert(self.model, createModel())
	end
	print(model)
	-- #
	-- three in a mask
	parent.__init(self)

	self.updateModel2self()

	self.reset()
end

function createModel()
	-- inputSize and be {N, 2}
	local submodel = {}
	submodel.input = torch.Tensor(input_pmodel)
	
	submodel.weight = torch.Tensor(input_pmodel, output_pmodel)
	submodel.gradWeight = torch.Tensor(input_pmodel, output_pmodel)
	
	submodel.bias = torch.Tensor(1) -- each model have one bias, intotal num_model
	submodel.gradBias = torch.Tensor(1) -- 
	
	submodel.output = torch.Tensor(output_pmodel)
	submodel.gradOutput = torch.Tensor(output_pmodel)
	
	return submodel
end

function PartialConnected:updateModel2self()
	self.weight = self.model[1].weight
	self.bias = self.model[1].bias
	self.gradWeight = self.model[1].gradWeight
	self.gradBias = self.model[1].gradBias

	for i = 2, num_model do
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
      stdv = 1./math.sqrt(self.model[1].weight:size(2))
   end

   for i = 1, num_model do
      self.model[i].weight:uniform(-stdv, stdv)
      self.model[i].bias:uniform(-stdv, stdv)
   end

   return self
end


function PartialConnected:updateOutput(input) 
	-- input size； BATCH * D
	-- input dimension should be 12 x 1 vector
	-- output = torch.Tensor(4):fill(0) -- initilaize
	if(not input_split_flag) then splitInput2Sub(input) end

	if(input:dim() ~= 1) then 
		-- reshape()
		for i = 1, #self.model do -- 1 to 4
			
			p(self.model[i].weight)
			p(self.model[i].input)
			p(self.model[i].bias)
			p(self.model[i].output)
			
			self.addbuffer = torch.Tensor(output_pmodel):fill(1):float() -- scala in this case
			self.model[i].output:addmm(self.model[i].weight * self.model[i].input)
			self.model[i].output:addr(1, self.model[i].output, 1, self.model[i].bias, self.addbuffer)
			-- use torch.addmv([res,] [beta,] [v1,] vec1, [v2,] mat, vec2)
			-- torch.addmv(1Db_vec(N,D), X(N,D), W(D))
			-- res = (beta * res) + (v1 * vec1) + (v2 * (mat * vec2))
			-- beta: momentumn, v1: bias, v2 = 1, mat: X, vec2: weight
			print(self.model[i].output)
		end
	
	else
		print('dimension error')
		-- sli = input:reshape(input:size(1),4,2) 
		-- for i = 1, #sli[1] do -- 1 to 4
		-- output[i] = bias[i] + weight[i] * sli[i] -- use torch.addmv
		-- output:addmv()
	end
	-- cat output
	local out = self.model[1].output

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

function PatialConnected:accGradParameters(input, gradOutput, scale)
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

		for i = 1, #self.model
			-- ?? self.weight[i].addr(scale, gradOutput[i], sli[i])

			self.model[i].gradWeight:addmm(scale, self.model[i].gradOutput:t(), self.model[i].input) 
			self.model[i].gradBias:addmv(scale, self.model[i].gradOutput:t(), self.model[i].addBuffer) 
			-- in the case of output is scala
	else 
		print('input dimension should be one')
	end

	self:updateModel2self()

end

function PatialConnected:updateGradInput(input, gradOutput)
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
			self.model[i].gradInput:addmv(0, 1, self.model[i].gradOutput, self.model[i].weight)
		end
	end

	self:updateModel2self()
end


function PartialConnected: splitInput2Sub(input)
	input_split_flag = true
	for i=1, #self.model do
		self.submodel[i].input = input[{{}, 
			{1 + (i-1) * #self.submodel[i].input, i * #self.submodel[i].input}}]

function PartialConnected:splitGradOutput2Sub(gradOutput)
	gradOutput_split_flag = true
	for i=1, #self.model do
		self.submodel[i].gradOutput = gradOutput[{{}, 
			{1 + (i-1) * #self.submodel[i].output, i * #self.submodel[i].output}}]

function p(x)
	print(p:size())
end
