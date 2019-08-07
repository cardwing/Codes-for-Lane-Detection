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
   assert(torch.max(label)<=6, 'label larger than 6!')
   assert(torch.min(label)>=0, 'label smaller than 0!')
   label:add(1)
   return {
      input = image,
      segLabel = label,
      exist = self.imageInfo.Exist[i],
      imgpath = self.imageInfo.imagePath[i],
   }
end

--[[function LaneDataset:get_seq(i, frame)
   local imgpath = ffi.string(self.imageInfo.imagePath[i+frame-1]:data())
   local lbpath = ffi.string(self.imageInfo.labelPath[i+frame-1]:data())
   local image = self:_loadImage(self.dir .. imgpath, 3, 'float')
   local label = self:_loadImage(self.dir .. lbpath, 1, 'byte')
   label:add(1)
   return {
      input = image,
      target = label,
      imgpath = self.imageInfo.imagePath[i+frame-1],
   }
end]]

function LaneDataset:_loadImage(path, channel, ttype)
   local ok, input = pcall(function()
      return image.load(path, channel, ttype)
   end)

   if not ok then
      print(path)
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
         --t.ScaleWH(768, 432),
         --t.ScaleWH(850, 368),
         t.ScaleWH(640, 368),
         --t.RandomCrop(640, 368),
         t.ColorNormalize(meanstd),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleWH(640, 368),
         --t.ScaleWH(768, 432),
         --t.ScaleWH(850, 368),
         --t.RandomCrop(640, 368),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

function LaneDataset:preprocess_aug()
   if self.split == 'train' then
   return t.Compose{
         --t.RandomScaleRatio(780, 864, 274, 304),
         t.RandomScaleRatio(860, 900, 425, 445),
         --t.RandomScaleRatio(860, 940, 412, 452),  --for r50 large
         --t.RandomScaleRatio(600, 680, 350, 390),
         t.ColorNormalize(meanstd),
         t.Rotation(1, -1.7),
         --t.RandomCrop(728, 264),
         t.RandomCropLane(640, 368),
         --t.RandomCropLane(768, 432),
         --t.RandomCrop(816, 288),
      }
   elseif self.split == 'val' then
      return t.Compose{
         t.ScaleWH(640, 368),
         --t.ScaleWH(880, 435),
         --t.ScaleWH(900, 432),
         --t.RandomCropLane(640, 368),
         --t.RandomCropLane(768, 432),
         t.ColorNormalize(meanstd),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.LaneDataset
