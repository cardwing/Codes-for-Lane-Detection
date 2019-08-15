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
   local image = self:_loadImage(self.dir .. imgpath, 3, 'float')
   --print(imgpath)
   return {
      input = image,
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

function LaneDataset:preprocess()
   if self.split == 'train' then
   return t.Compose{
         --t.Min(),
         --t.ScaleWH(640, 368),
         t.ScaleWH(1280, 720),
         t.ColorNormalize(meanstd),
      }
   elseif self.split == 'val' then
      return t.Compose{
         --t.Min(),
         t.ScaleWH(1280, 720),
         --t.ScaleWH(800, 288),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

function LaneDataset:preprocess_aug()
   if self.split == 'train' then
   return t.Compose{
         t.RandomScaleRatio(735, 898, 267, 326),
         t.Rotation(2),
         t.RandomCrop(728, 264),
         t.ColorNormalize(meanstd),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleWH(800, 288),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LaneDataset
