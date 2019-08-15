local paths = require 'paths'
local ffi = require 'ffi'

local M = {}

local function getPaths(file)
   local imagePath = torch.CharTensor()
   local labelPath = torch.CharTensor()

   local f = io.open(file, 'r')
   print('load file: ' .. file)
   local imMaxLength = -1
   local lbMaxLength = -1
   local imagePaths = {}
   local labelPaths = {}
   local exists = {}
   while true do
      local line = f:read()
      if line == nil then break end
      local lineSplit = line:split(' ')
      local impath = lineSplit[1]
      local lbpath = lineSplit[2]
      table.insert(imagePaths, impath)
      table.insert(labelPaths, lbpath)
      imMaxLength = math.max(imMaxLength, #impath + 1)
      lbMaxLength = math.max(lbMaxLength, #lbpath + 1)
      local exist = torch.Tensor(4):zero()
      for i = 1, 4 do
         exist[i] = lineSplit[i+2]
      end
      table.insert(exists, exist)
   end
   f.close()
   
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, imMaxLength):zero()
   local labelPath = torch.CharTensor(nImages, lbMaxLength):zero()
   local Exist = torch.Tensor(nImages, 4):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end
   for i, path in ipairs(labelPaths) do
      ffi.copy(labelPath[i]:data(), path)
   end
   for i, data in ipairs(exists) do
      Exist[i]:copy(data)
   end
   print("finish getPath")
   return imagePath, labelPath, Exist
end

local function getPerm(paths, seqLen, split)   -- Permute data list order
   local size = paths:size(1)
   local perm
   if split == 'train' then 
      perm = torch.randperm(size)
   elseif split == 'val' then
      perm = torch.Tensor(size)
      for i = 1, size do
         perm[i] = i
      end
   else
      print('Wrong split: ' .. split)
   end
   local seqPerm = {}
   for i = 1, size do
      local id = perm[i]
      if id <= size - seqLen + 1 then
         local function videoName(impath)
            local impath = string.sub(impath,2,-1)
            local j = string.find(impath, '/')
            if j then
               local video = string.sub(impath, j+1, -11)
               return video
            else
               return nil
            end
         end
         local impath = ffi.string(paths[id]:data())
         local impath2 = ffi.string(paths[id+seqLen-1]:data())
         local video = videoName(impath)
         local video2 = videoName(impath2)
         if video == video2 then
            table.insert(seqPerm, id)
         end
      end
   end
   local seqPermTensor = torch.Tensor(#seqPerm)
   for i = 1, #seqPerm do
      seqPermTensor[i] = seqPerm[i]
   end
   return seqPermTensor
end
   

function M.exec(opt, cacheFile)
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local labelPath = torch.CharTensor()  -- path to each label

   local listTrain = opt.train
   local listVal = opt.val
   
   local trainImagePath, trainLabelPath, trainExist = getPaths(listTrain)
   local valImagePath, valLabelPath, valExist = getPaths(listVal)
   local trainPerm = getPerm(trainImagePath, opt.seqLen, 'train')  -- set seqLen to 1 if you don't use rnn-based model
   local valPerm = getPerm(valImagePath, 1, 'val')
   print("create info")
   local info = {
      basedir = opt.data,
      train = {
         imagePath = trainImagePath,
         labelPath = trainLabelPath,
         Exist = trainExist,
         perm = trainPerm,
      },
      val = {
         imagePath = valImagePath,
         labelPath = valLabelPath,
         Exist = valExist,
         perm = valPerm,
      },
   }
   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
