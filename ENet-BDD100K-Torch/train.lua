--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local models = require 'models/init'
local checkpoints = require 'checkpoints'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState, checkpoint)
   print('init trainer')
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.iter = 1
   if checkpoint then
      self.iter = checkpoint.iter
   end
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   self.finish = false
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   print('training')

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum, lossSum2 = 0.0, 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      if self.iter>=self.opt.maxIter then
         self.finish = true
         break
      end
      self.optimState.learningRate = self:learningRate(epoch)
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      self.input = self.input:cuda()
      local output = self.model:forward(self.input)
      --print(output)
      local batchSize = output[1]:size(1)
      local loss, Loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)
      optim.sgd(feval, self.params, self.optimState)
      N = N + batchSize
      lossSum = lossSum + Loss[1]*batchSize    -- loss for segmentation branch
      lossSum2 = lossSum2 + Loss[2]*batchSize    -- loss for classification branch
      print((' | Epoch: [%d][%d/%d][%d]  Time %.2f  LR %.5f  Err1 %.5f (%.5f)  Err2 %.5f (%.5f)'):format(
            epoch, n, trainSize, self.iter, timer:time().real, self.optimState.learningRate, Loss[1], lossSum / N, Loss[2], lossSum2 / N))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      if self.iter % 500 == 0 then
         checkpoints.save(epoch, self.model, self.optimState, false, self.opt, self.iter)
      end
      timer:reset()
      dataTimer:reset()
      self.iter = self.iter + 1
   end

   return lossSum / N, self.finish
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local AccSum, RecSum, IOUSum, lossSum, lossSum2 = 0.0, 0.0, 0.0, 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      self.input = self.input:cuda()
      local output = self.model:forward(self.input)
      local accuracy, avgRecall, avgIOU
      local batchSize = 0
      --print(output[2]:size())
      --print(output[1]:size())
      batchSize = output[1]:size(1)
      accuracy, avgRecall, avgIOU = self:computeAccuracy(output[1]:float(), self.target[1]:float())
      AccSum = AccSum + accuracy*batchSize
      RecSum = RecSum + avgRecall*batchSize
      IOUSum = IOUSum + avgIOU*batchSize

      --print(self.target[1])
      --print(output)
      local loss, Loss = self.criterion:forward(self.model.output, self.target)
      N = N + batchSize
      lossSum = lossSum + Loss[1]*batchSize
      lossSum2 = lossSum2 + Loss[2]*batchSize
      print((' | Test: [%d][%d/%d] Err1 %.5f (%.5f) Err2 %.5f (%.5f) Acc %.2f (%.3f) mRec %.2f (%.3f) mIOU %.2f (%.3f)'):format(
         epoch, n, size, Loss[1], lossSum / N, Loss[2], lossSum2 / N, accuracy, AccSum / N, avgRecall, RecSum / N, avgIOU, IOUSum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   return lossSum / N, AccSum / N, RecSum / N, IOUSum / N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.segLabel = self.segLabel or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
   self.segLabel:resize(sample.target[1]:size()):copy(sample.target[1])
   self.exist = self.exist or torch.CudaLongTensor()
   self.exist:resize(sample.target[2]:size()):copy(sample.target[2])
   self.target = {self.segLabel:cuda(), self.exist:cuda()}
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'lane' then
      decay = 1 - self.iter/self.opt.maxIter
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(decay, 0.9)
end

function Trainer:computeAccuracy( output, target )
   -- This is not the final evaluation code.
   -- This only gives primal evaluation for segmentation.
   local batchSize = output:size(1)
   local classNum = output:size(2)
   local h = output:size(3)
   local w = output:size(4)
   local accuracy, avgRecall, avgIOU = 0.0, 0.0, 0.0

   local lane_accuracy = 0.0

   for i = 1, batchSize do
      local _, maxMap = torch.max(output[{i,{},{},{}}], 1)
      local target_i = target[{i,{},{}}]:long()
      -- accuracy

      -- accuracy = accuracy + torch.sum(torch.eq(maxMap, target_i)) / (h*w)
      -- print(target_i)
      
      if torch.sum(target_i - torch.LongTensor(720, 1280):fill(1)) ~= 0 then
         accuracy = accuracy + torch.sum(torch.cmul(target_i - torch.LongTensor(720, 1280):fill(1), maxMap - torch.LongTensor(720, 1280):fill(1))) / torch.sum(target_i - torch.LongTensor(720, 1280):fill(1))
      end    

      -- lane_accuracy = lane_accuracy + torch.sum(torch.cmul(target_i, maxMap)) / torch.sum(target_i)

      -- recall, IOU
      local recall = 0.0
      local IOU = 0.0
      local numClass, numUnion = 0, 0
      for c = 2, classNum do
         local num_c = torch.sum(torch.eq(target_i, c))
         local num_c_pred = torch.sum(torch.eq(maxMap, c))
         local numTrue = torch.sum(torch.cmul(torch.eq(maxMap, c), torch.eq(target_i, c)))
         local unionSize = num_c + num_c_pred - numTrue
         if num_c > 0 or num_c_pred > 0 then
            IOU = IOU + numTrue / unionSize
            numUnion = numUnion + 1
         end
         if num_c > 0 then
            recall = recall + numTrue / num_c
            numClass = numClass + 1
         end
      end
      if numClass ~= 0 then
         recall = recall / numClass
      else
         recall = 0
      end
      avgRecall = avgRecall + recall
      if numUnion ~= 0 then
         IOU = IOU / numUnion
      else
         IOU = 0
      end
      avgIOU = avgIOU + IOU
   end
   accuracy = accuracy / batchSize
   avgRecall = avgRecall / batchSize
   avgIOU = avgIOU / batchSize
   return accuracy * 100, avgRecall * 100, avgIOU * 100
end

return M.Trainer
