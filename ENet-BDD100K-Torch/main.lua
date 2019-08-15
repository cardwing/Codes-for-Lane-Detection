--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local opts = require 'opts'
local opt = opts.parse(arg)
local DataLoader = require 'dataloader'
local Trainer = require 'train'
local models = require 'models/init'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState, checkpoint)

---- Have a test ----

-- opt.testOnly = true

if opt.testOnly then
   local Err, Acc, Rec, IOU = trainer:test(0, valLoader)
   print(string.format(' * Results: Err: %.3f Acc: %.3f Rec: %.3f IOU: %.3f', Err, Acc, Rec, IOU))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestLoss = math.huge
if checkpoint then
   bestLoss = bestLoss or checkpoint.bestLoss
end
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss, finish = trainer:train(epoch, trainLoader)
   print(string.format(' * TrainLoss: %.3f', trainLoss))
   -- Run model on validation set
   local valLoss, Acc, Rec, IOU = 0.0, 0.0, 0.0, 0.0
   valLoss, Acc, Rec, IOU = trainer:test(epoch, valLoader)

   local bestModel = false
   if valLoss < bestLoss then
      bestModel = true
      bestLoss = valLoss
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt, trainer.iter, bestLoss)
   if finish then
      break
   end
end

print(string.format(' * Finished Err: %6.3f', bestLoss))
