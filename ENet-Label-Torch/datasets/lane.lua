local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local LaneDataset = torch.class('resnet.LaneDataset', M)

function LaneDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function LaneDataset:get(i)
   local imgpath = ffi.string(self.imageInfo.imagePath[i]:data())
   local lbpath = ffi.string(self.imageInfo.labelPath[i]:data())
   local image = self:_loadImage(self.dir .. imgpath, 3, 'float')
   local label = self:_loadImage(self.dir .. lbpath, 1, 'byte')
   label:add(1)
   return {
      input = image,
      segLabel = label,
      exist = self.imageInfo.Exist[i],
      imgpath = self.imageInfo.imagePath[i],
   }
end

function LaneDataset:_loadImage(path, channel, ttype)
   local ok, input = pcall(function()
      return image.load(path, channel, ttype)
   end)

   if not ok then
      print("load image failed!")
      return -1
   end
   return input
end

function LaneDataset:size()
   return self.imageInfo.imagePath:size(1)
end

local meanstd = {
   mean = { 0.3598, 0.3653, 0.3662 },
   std = { 0.2573, 0.2663, 0.2756 },
}

function LaneDataset:preprocess()           -- Don't use data augmentation for training RNN
   if self.split == 'train' then
   return t.Compose{
         t.ScaleWH(640, 368),
         t.ColorNormalize(meanstd),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleWH(640, 368),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

function LaneDataset:preprocess_aug()
   if self.split == 'train' then
   return t.Compose{
         t.RandomScaleRatio(936, 1018, 194, 224), -- 760, 842, 274, 304
         t.ColorNormalize(meanstd),
         t.Rotation(2),
         --t.RandomCrop(800, 288),
         t.RandomCropLane(976, 208), -- 800, 288
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleWH(976, 208), -- 800, 288
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LaneDataset
