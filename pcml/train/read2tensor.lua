--[[usega: dofile 'read2tensor.lua'
load x and y into memory, 
x: 6000x36865 tensor, y: 6000x1 tensor

]]--


require 'io'
require 'torch'


local function split(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end


file = io.open('cnnft.csv','r')

i = 1
x = torch.Tensor(6000,36865)
temp = torch.Tensor(36865)
for line in file:lines() do
    -- print(type(line))
    col = split(line, ',')
    -- print(type(col))
    -- print(table.getn(col))
    for k = 1,36865 do 
    --    print(k)
	x[i][k] = tonumber(col[k])
    end
    -- print(2)
    -- x[i] = k
    i = i + 1
end

file:close()

label = io.open('label.csv','r')
i = 1
y = torch.Tensor(6000)
for line in label:lines() do
    y[i] = tonumber(line)
    i = i + 1
end

label:close()
