--[[this file is to read the png file in folder, with the label.csv(for training set) and convert them into .bin filey --]]
require 'image'
require '../kaggle-cifar10-torch7/SETTINGS'
DATA_DIR = './train'
string.split_it = function(str, sep)
   if str == nil then return nil end
   return string.gmatch(str, "[^\\" .. sep .. "]+")
end


string.split = function(str, sep)

   local ret = {}

   for seg in string.split_it(str, sep) do
      ret[#ret+1] = seg
   end

   return ret
end


local function label_vector(label_name)

   local vec = torch.Tensor(10):zero()

   vec[LABEL2ID[label_name]] = 1.0

   return vec
end

local TRAIN_N = 6000

local function convert_train()
   local label_file = string.format("%s/label.csv", DATA_DIR)
   local x = torch.Tensor(TRAIN_N, 3, 231, 231)
   local y = torch.Tensor(TRAIN_N, 4)
   local file = io.open(label_file, "r")
   local head = true
   local line
   local i = 1
   for line in file:lines() do
      -- the first line is head, ignore it
      if head then
	 head = false
      else
	 -- file format: id, labelstry
	 -- local col = string.split(line, ",")
	 -- img is torch.DoubleTensor of size 3x32x32
         local id =  ("%04d"):format( i )

	 
         local imageFile = string.format("%s/imgs/train0%s.jpg", DATA_DIR, id)
         print(imageFile)
         local img = image.load(imageFile)
	 x[i]:copy(img)
	 --y[i]:copy(label_vector(col[2]))
         y[i] = tonumber(line)
	 -- (x, y) form a training data sample pair

	 if i % 100 == 0 then
	    xlua.progress(i, TRAIN_N)
	 end

	 i = i + 1
      end
   end
   file:close()
   
   torch.save(string.format("%s/train_x.bin", DATA_DIR), x)
   torch.save(string.format("%s/train_y.bin", DATA_DIR), y)
end
local TEST_N = 300000

local function convert_test()
   local x = torch.Tensor(TEST_N, 3, 32, 32)
   local i = 1
   for i = 1, TEST_N do
      local img = image.load(string.format("%s/test/%d.png", DATA_DIR, i))
      x[i]:copy(img)
      if i % 100 == 0 then
	 xlua.progress(i, TEST_N)
      end
   end
   torch.save(string.format("%s/test_x.bin", DATA_DIR), x)
end

print("convert train data ...")
convert_train()
print("bug with test data do not convert test data ...")
-- convert_test()
