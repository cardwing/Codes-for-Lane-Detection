local paths = require 'paths'
local ffi = require 'ffi'

local M = {}

local function getPaths(file)
   local imagePath = torch.CharTensor()

   local f = io.open(file, 'r')
   print('load file: ' .. file)
   local imMaxLength = -1
   local imagePaths = {}
   while true do
      local line = f:read()
      if line == nil then break end
      
      local impath = line
      table.insert(imagePaths, impath)
      imMaxLength = math.max(imMaxLength, #impath + 1)
   end
   f.close()
   
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, imMaxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end
   print("finish getPath")
   return imagePath
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

   local listVal = opt.val
   
   local valImagePath = getPaths(listVal)
   local valPerm = getPerm(valImagePath, 1, 'val')
   print("create info")
   local info = {
      basedir = opt.data,
      train = nil,
      val = {
         imagePath = valImagePath,
         perm = valPerm,
      },
   }
   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
