--[[read csv file. 
if mat file, use matlab 'csvwrite' convert to csv file first

usega: dofile 'read2tensor.lua'
load x and y into memory, 
x: 6000x36865 tensor, y: 6000x1 tensor-- for cnnft.csv
x: 6000x5408 tensor, y: 6000x1 tensor -- for hogft.csv

]]--

info = lapp [[
  -c, --csv  (default "cnnft.csv") datafile in csv
]]

require 'io'
require 'torch'
local ft_dim = 36864

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

file = io.open(info.csv,'r')
-- file = io.open('cnnft.csv','r')

i = 1
-- x = torch.Tensor(6000,36865)
-- temp = torch.Tensor(36865)

x = torch.Tensor(6000,ft_dim)
temp = torch.Tensor(x:size()[2])

for line in file:lines() do
    -- print(type(line))
    col = split(line, ',')
    -- print(type(col))
    -- print(table.getn(col))
    for k = 1,x:size()[2] do 
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
y = torch.Tensor(x:size()[1])

for line in label:lines() do
    y[i] = tonumber(line)
    i = i + 1
end

label:close()

-- save file as 'bin'
filename = string.split(info.csv,'csv')
torch.save('Trainset_x_'..filename[1]..'bin',x)
torch.save('Trainset_y_'..filename[1]..'bin',y)

