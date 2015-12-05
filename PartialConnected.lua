local Linear, parent = torch.class('nn.PatialConnected', 'nn.Module')

function PatialConnected: __init(inputSize, outputSize)
	inputSize = 8
	outputsize = 4
	-- three in a mask
	parent.__init(self)

	self.weight = {torch.Tensor(1,2),torch.Tensor(1,2), torch.Tensor(1,2), torch.Tensor(1,2)}
	self.bias = torch.Tensor(4)
	self.gradWeight = {torch.Tensor(1,2),torch.Tensor(1,2), torch.Tensor(1,2), torch.Tensor(1,2)}
	self.gradBias = torch.Tensor(4)

function PatialConnected:updateOutput(input) 
	-- input size； BATCH * D
	-- input dimension should be 12 x 1 vector
	output = torch.Tensor(4):fill(0) -- initilaize

	if(input:dim()==1) then {
		-- reshape()
		local sli = input:reshape(4,2) 
		local output = torch.Tensor(4):fill(0) -- initilaize
		for i = 1, #sli[1] do -- 1 to 4
			output[i] = bias[i] + weight[i] * sli[i] -- use torch.addmv
		
		print(output)
		self.output = output
	}
	else{
		-- sli = input:reshape(input:size(1),4,2) 
		-- for i = 1, #sli[1] do -- 1 to 4
		-- output[i] = bias[i] + weight[i] * sli[i] -- use torch.addmv
		-- output:addmv()
	}
	return self.output -- 4x1
end

function PatialConnected:backward(intput, gradOutput)
	--[[bp, gradOutput
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
	--]] 


end

function PatialConnected:accGradParameters(input, gradOutput, scale)
	--[[ 
		gradient wrt to W:
			= GET * \patial{output}/\partial{para} 
			= GET * X 
			= input * gradOutput

		W' = W - step * gradient // tradictional batch gradient descend
		W' = momentunm*W - lr * gradient // 
		
		gradient wrt to bias:
			= GET * \partial{output = W^T + b}/\partial{b}
			= GET
		bias = bias - step * GET
	--]]

	if(input:dim()==1) then {
		-- check dimention: reshape()
		sli = input:reshape(4,2) 
		-- gradOutput in shape(#current layer output, 1)
		-- assume scale, is the same for all model
		for i = 1, #sli[1]
			self.weight[i].addr(scale, gradOutput[i], sli[i])
			self.gradBias[i].add(scale, gradOutput[i]) 
	}else 
		print('input dimension should be one')
end

function PatialConnected:updateGradInput(input, gradOutput)
	--[[ compute gradient of the module wrt its own input
		return: gradInput, 4 model, each in dimension 2 + 1, 
		4 X 3 dimension 
		
		pass to the previous layer: 
		PASS 
			= GET * \partial{output}/\partial{input}
			= GET * W
	--]]
	self.gradInput = torch.Tensor
	if input:dim() == 1 then
		for i = 1,4
		self.gradInput = torch.addmv(0, 1, self)

end


