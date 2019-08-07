--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      print("data created")
      loaders[i] = M.DataLoader(dataset, opt, split)
      print("data loaded")
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      _G.preprocess_aug = dataset:preprocess_aug()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   -- self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.nCrops = 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.split = split
   self.labelType = opt.labelType
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local labelType = self.labelType
   --if self.split == 'val' then
      --batchSize = torch.round(batchSize / 2)
   --end
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1)
               local batch, segLabels, exists, imgpaths
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  --print(sample.input:size())
                  --print(sample.segLabel)
                  local input, segLabel, exist
                  if sample.input:size(3) == 1640 or sample.input:size(3) == 1280 then
                     input, segLabel, exist = _G.preprocess_aug(sample.input, sample.segLabel, sample.exist)
                  else
                     input, segLabel, exist = _G.preprocess(sample.input, sample.segLabel, sample.exist)
                  end
                  segLabel:resize(segLabel:size(2),segLabel:size(3))
                  if not batch then
                     local imageSize = input:size():totable()
                     local labelSize = segLabel:size():totable()
                     local existSize = exist:size():totable()
                     local pathSize = sample.imgpath:size():totable()
                     --if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch.FloatTensor(sz, table.unpack(imageSize))
                     segLabels = torch.FloatTensor(sz, table.unpack(labelSize))
                     exists = torch.FloatTensor(sz, table.unpack(existSize))
                     imgpaths = torch.CharTensor(sz, table.unpack(pathSize))
                  end
                  batch[i]:copy(input)
                  segLabels[i]:copy(segLabel)
                  exists[i]:copy(exist)
                  imgpaths[i]:copy(sample.imgpath)
               end
               local targets
               if labelType=='seg' then
                  targets = segLabels
               elseif labelType=='exist' then
                  targets = exists
               elseif labelType=='segExist' then
                  targets = {segLabels, exists}
               else
                  cmd:error('unknown labelType: ' .. labelType)
               end
               collectgarbage(); collectgarbage()

               return {
                  input = batch,
                  target = targets,
                  imgpath = imgpaths, -- used in test
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end
   return loop
end

return M.DataLoader
