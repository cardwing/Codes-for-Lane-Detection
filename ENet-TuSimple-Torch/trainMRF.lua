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
   if opt.modelPart1 ~= 'none' or opt.netType == 'MRF' or opt.netType == 'CRF' or opt.netType == 'ReNet' or (opt.netType == 'regNet' and opt.labelType ~= 'segReg') then
      opt.initCnn = true
      self.part1 = true
      local cnn = models.setup(opt)
      cnn:evaluate()
      self.cnn = cnn
   end
   self.id = 2
   if opt.labelType == 'segReg' then
      self.id = 3
   end
   --local learningRates, weightDecays = model:getOptimConfig(opt.LR, opt.weightDecay)
   self.optimState = optimState or {
   --   learningRates = learningRates,
   --   weightDecays = weightDecays,
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
   --   self.optimState.learningRates, self.optimState.weightDecays = self.model:getOptimConfig(self.optimState.learningRate, self.optimState.weightDecay)
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      self.input = self.input:cuda()
      --self.target = self.target:cuda()
      if self.part1 then
         self.input2 = self.cnn:forward(self.input)
      else
         self.input2 = self.input
      end
      self.input2 = self.input
      -- print(self.input2)

      local output = self.model:forward(self.input2)
      --print(output)
      local batchSize = 0
      if self.opt.labelType == 'seg' or self.opt.labelType == 'exist' then
         batchSize = output:size(1)
      else
         batchSize = output[1]:size(1)
      end
      --print(self.target[1])
      --tmp = self.target[1]:clone()
      --table.sort(tmp)
      --print(tmp[#tmp])
      local loss, Loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)
      optim.sgd(feval, self.params, self.optimState)
      N = N + batchSize
      if self.opt.labelType == 'seg' or self.opt.labelType == 'exist' then
         lossSum = lossSum + loss*batchSize
         print((' | Epoch: [%d][%d/%d][%d]  Time %.3f  Data %.3f  LR %.5f  Err %.4f (%.4f)'):format(
            epoch, n, trainSize, self.iter, timer:time().real, dataTime, self.optimState.learningRate, loss, lossSum / N))
      else
         lossSum = lossSum + Loss[1]*batchSize
         lossSum2 = lossSum2 + Loss[self.id]*batchSize
         print((' | Epoch: [%d][%d/%d][%d]  Time %.2f  LR %.5f  Err %.5f (%.5f)  Err %.5f (%.5f)'):format(
            epoch, n, trainSize, self.iter, timer:time().real, self.optimState.learningRate, Loss[1], lossSum / N, Loss[self.id], lossSum2 / N))
      end

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())
      --if self.iter % 500 == 0 then
         --checkpoints.save(epoch, self.model, self.optimState, false, self.opt, self.iter)
      --end
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
      --self.target = self.target:cuda()
      if self.part1 then
         self.input2 = self.cnn:forward(self.input)
      else
         self.input2 = self.input
      end
      -- print(self.input2)
      -- print(self.target)
      self.input2 = self.input
      local output = self.model:forward(self.input2)
      local accuracy, avgRecall, avgIOU
      local batchSize = 0
      if self.opt.labelType == 'seg' then
         batchSize = output:size(1)
         accuracy, avgRecall, avgIOU = self:computeAccuracy(output:float(), self.target:float())
         AccSum = AccSum + accuracy*batchSize
         RecSum = RecSum + avgRecall*batchSize
         IOUSum = IOUSum + avgIOU*batchSize
      elseif self.opt.labelType == 'segExist' then
         batchSize = output[1]:size(1)
         accuracy, avgRecall, avgIOU = self:computeAccuracy(output[1]:float(), self.target[1]:float())
         AccSum = AccSum + accuracy*batchSize
         RecSum = RecSum + avgRecall*batchSize
         IOUSum = IOUSum + avgIOU*batchSize
      elseif self.opt.labelType == 'exist' then
         batchSize = output:size(1)
      else
         batchSize = output[1]:size(1)
      end
      local loss, Loss = self.criterion:forward(self.model.output, self.target)
      N = N + batchSize
      if self.opt.labelType == 'seg' then
         lossSum = lossSum + loss*batchSize
         print((' | Test: [%d][%d/%d] Err %.3f  Acc %.2f (%.3f)  mRec %.2f (%.3f)  mIOU %.2f (%.3f)'):format(
            epoch, n, size, loss, accuracy, AccSum / N, avgRecall, RecSum / N, avgIOU, IOUSum / N))
      elseif self.opt.labelType == 'exist' then
         lossSum = lossSum + loss*batchSize
         print((' | Test: [%d][%d/%d] Err %.5f (%.5f)'):format(
            epoch, n, size, loss, lossSum / N))
      elseif self.opt.labelType == 'segExist' then
         lossSum = lossSum + Loss[1]*batchSize
         lossSum2 = lossSum2 + Loss[self.id]*batchSize
         print((' | Test: [%d][%d/%d] Err %.5f (%.5f) Err %.5f (%.5f) Acc %.2f (%.3f) mRec %.2f (%.3f) mIOU %.2f (%.3f)'):format(
            epoch, n, size, Loss[1], lossSum / N, Loss[self.id], lossSum2 / N, accuracy, AccSum / N, avgRecall, RecSum / N, avgIOU, IOUSum / N))
      else
         lossSum = lossSum + Loss[1]*batchSize
         lossSum2 = lossSum2 + Loss[self.id]*batchSize
         print((' | Test: [%d][%d/%d] Err %.5f (%.5f) Err %.5f (%.5f)'):format(
            epoch, n, size, Loss[1], lossSum / N, Loss[self.id], lossSum2 / N))
      end

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
   if self.opt.labelType == 'seg' then
      self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
      self.target:resize(sample.target:size()):copy(sample.target):cuda()
   elseif self.opt.labelType == 'exist' then
      self.target = self.target or torch.CudaTensor()
      self.target:resize(sample.target:size()):copy(sample.target):cuda()
   elseif self.opt.labelType == 'reg' then
      self.exist = self.exist or torch.CudaLongTensor()
      self.coordinate = self.coordinate or torch.CudaTensor()
      self.exist:resize(sample.target[1]:size()):copy(sample.target[1])
      self.coordinate:resize(sample.target[2]:size()):copy(sample.target[2])
      self.target = {self.exist:cuda(), self.coordinate:cuda()}
   elseif self.opt.labelType == 'segExist' then
      self.segLabel = self.segLabel or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
      self.segLabel:resize(sample.target[1]:size()):copy(sample.target[1])
      self.exist = self.exist or torch.CudaLongTensor()
      self.exist:resize(sample.target[2]:size()):copy(sample.target[2])
      self.target = {self.segLabel:cuda(), self.exist:cuda()}
   elseif self.opt.labelType == 'segReg' then
      self.segLabel = self.segLabel or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
      self.segLabel:resize(sample.target[1]:size()):copy(sample.target[1])
      self.exist = self.exist or torch.CudaLongTensor()
      self.coordinate = self.coordinate or torch.CudaTensor()
      self.exist:resize(sample.target[2]:size()):copy(sample.target[2])
      self.coordinate:resize(sample.target[3]:size()):copy(sample.target[3])
      self.target = {self.segLabel:cuda(), self.exist:cuda(), self.coordinate:cuda()}
   end
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'lane' or self.opt.dataset == 'laneE' or self.opt.dataset == 'laneReg' then
      decay = 1 - self.iter/self.opt.maxIter
   --elseif self.opt.dataset == 'laneReg' then
      --local step = math.floor(self.iter/5000)
      --return self.opt.LR * math.pow(0.5, step)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(decay, 0.9)
end

function Trainer:computeAccuracy( output, target )
   local batchSize = output:size(1)
   local classNum = output:size(2)
   local h = output:size(3)
   local w = output:size(4)
   local accuracy, avgRecall, avgIOU = 0.0, 0.0, 0.0
   for i = 1, batchSize do
      local _, maxMap = torch.max(output[{i,{},{},{}}], 1)
      local target_i = target[{i,{},{}}]:long()
      -- accuracy
      accuracy = accuracy + torch.sum(torch.eq(maxMap, target_i)) / (h*w)
      -- recall, IOU
      local recall = 0.0
      local IOU = 0.0
      local numClass, numUnion = 0, 0
      for c = 1, classNum do
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
      recall = recall / numClass
      avgRecall = avgRecall + recall
      IOU = IOU / numUnion
      avgIOU = avgIOU + IOU
   end
   accuracy = accuracy / batchSize
   avgRecall = avgRecall / batchSize
   avgIOU = avgIOU / batchSize
   return accuracy * 100, avgRecall * 100, avgIOU * 100
end

return M.Trainer
