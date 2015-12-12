dofile 'PartialConnected.lua'


input = torch.Tensor(10, 6, 1,1 ):uniform(-1,1)
input:float()
net = nn.PartialConnected(10, 6, 2) -- batch 10, input 6, output 2, 3 model, input_pmodel = 2
net:float()
output = net:forward(input)
print(output)

df_do = input.new():resizeAs(input):uniform(1,2):float()
df_do:float()
net:backward(input, df_do)
print(net:parameters())
print(net.gradInput)


