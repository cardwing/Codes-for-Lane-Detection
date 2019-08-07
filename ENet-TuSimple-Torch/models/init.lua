--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
--require 'rnn'
--require 'PixelDeshuffle'
--require 'SpatialConvolutionH'
--require 'ConvLSTM'
--require 'HoleConvLSTM'
--require 'ConvGRU'
--require 'PixelSplit'
--require 'PixelMerge'
--require 'TensorSplit'
--require 'TensorMerge'
require 'ParallelCriterion2'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath)
      if opt.softmax then
         model:add(cudnn.SpatialSoftMax())
         print('Softmax added')
      end
      model:cuda()
   --[[elseif opt.initCnn and opt.modelPart1 ~= 'none' then
      assert(paths.filep(opt.modelPart1), 'Pretrained CNN not found: ' .. opt.cnnPath)
      print('=> Loading pretrained CNN from ' .. opt.cnnPath)
      model = torch.load(opt.modelPart1)
      model:cuda()
   elseif opt.initCnn and opt.cnnPath ~= 'none' then                -- For process image so as to train RNN part only
      assert(paths.filep(opt.cnnPath), 'Pretrained CNN not found: ' .. opt.cnnPath)
      print('=> Loading pretrained CNN from ' .. opt.cnnPath)
      model = torch.load(opt.cnnPath)
      if opt.netType == 'MRF' or opt.netType == 'MRF2'  or opt.netType == 'ReNet' then
         model:remove()
         model:remove()
         model:remove()
         --model:add(cudnn.SpatialSoftMax())
      elseif opt.netType == 'regNet' then
         model:remove()
         model:add(cudnn.SpatialSoftMax())
      elseif opt.netType ~= 'CRF' then
         model:get(1).gradInput = torch.CudaTensor()
         model:remove()
         model:remove()
         model:remove()
         if opt.nFTConv > 0 then
            for i = 1, opt.nFTConv*3 do  -- Should be modified if nFTConv>6
               model:remove()
            end
         end
         model = nn.Sequencer(model)
      end
      model:cuda()]]--
   elseif opt.retrain ~= 'none' then                                -- For fine tuning CNN
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
      model.__memoryOptimized = nil
   --[[elseif opt.MRFmodel ~= 'none' and opt.model ~= 'none' then
      assert(paths.filep(opt.model), 'File not found: ' .. opt.model)
      assert(paths.filep(opt.MRFmodel), 'File not found: ' .. opt.MRFmodel)
      print('Loading model from file: ' .. opt.model)
      model = torch.load(opt.model):cuda()
      print('Loading model from file: ' .. opt.MRFmodel)
      local MRF = torch.load(opt.MRFmodel):cuda()
      if opt.netType == 'regNet' then
         model:remove()
         model:add(cudnn.SpatialSoftMax())
         model:add(MRF)
      else
         model:remove()
         model:remove()
         model:remove()
         --model:add(cudnn.SpatialSoftMax())
         model:add(MRF)
         model:add(cudnn.SpatialSoftMax())
      end
      if opt.smooth then
         local avg = nn.Sequential()
         avg:add(nn.SplitTable(1, 3))
         avg:add(nn.NarrowTable(2, 4))
         local paral = nn.ParallelTable()
         local seq = nn.Sequential()
         seq:add(nn.Contiguous())
         seq:add(nn.View(1, 288, 816):setNumInputDims(2))
         local conv = cudnn.SpatialConvolution(1, 1, 9, 9, 1, 1, 4, 4)
         conv.weight:fill(1/81)
         conv.bias:fill(0)
         seq:add(conv)
         paral:add(seq)
         for i=1, 3 do
            paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
         end
         avg:add(paral)
         avg:add(nn.JoinTable(1, 3))
         model:add(avg:cuda())
      end
      model:cuda()]]--
   elseif opt.model ~= 'none' then                                 -- For testing CNN
      assert(paths.filep(opt.model), 'File not found: ' .. opt.model)   
      print('Loading model from file: ' .. opt.model)
      model = torch.load(opt.model):cuda()
      --[[if opt.dataset ~= 'coord' then
         model:add(cudnn.SpatialSoftMax())
      end
      if opt.smooth then
         local avg = nn.Sequential()
         avg:add(nn.SplitTable(1, 3))
         avg:add(nn.NarrowTable(2, 4))
         local paral = nn.ParallelTable()
         local seq = nn.Sequential()
         seq:add(nn.Contiguous())
         seq:add(nn.View(1, 288, 816):setNumInputDims(2))
         local conv = cudnn.SpatialConvolution(1, 1, 9, 9, 1, 1, 4, 4)
         conv.weight:fill(1/81)
         conv.bias:fill(0)
         seq:add(conv)
         paral:add(seq)
         for i=1, 3 do
            paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
         end
         avg:add(paral)
         avg:add(nn.JoinTable(1, 3))
         model:add(avg:cuda())
      end
      model:cuda()]]
   elseif opt.cnnPath ~= 'none' and opt.rnnPath ~= 'none' then      -- For testing CNN-RNN network
      local cnn, Rnn
      assert(paths.filep(opt.cnnPath), 'Pretrained CNN not found: ' .. opt.cnnPath)
      print('=> Loading pretrained CNN from ' .. opt.cnnPath)
      cnn = torch.load(opt.cnnPath)
      cnn:get(1).gradInput = torch.CudaTensor()
      cnn:remove()
      cnn:remove()
      cnn:remove()
      if opt.nFTConv > 0 then
         for i = 1, cnn:size()-opt.nFTConv*3 do  -- Should be modified if nFTConv>6
            cnn:remove(1)
         end
      end
      cnn = nn.Sequencer(cnn)
      assert(paths.filep(opt.rnnPath), 'Pretrained RNN not found: ' .. opt.rnnPath)
      print('=> Loading pretrained RNN from ' .. opt.rnnPath)
      Rnn = torch.load(opt.rnnPath)
      print(Rnn)
      model = nn.Sequential()
      model:add(cnn)
      model:add(Rnn)
      model:add(nn.Sequencer(cudnn.SpatialSoftMax()))
      if opt.smooth then
         local avg = nn.Sequential()
         avg:add(nn.SplitTable(1, 3))
         avg:add(nn.NarrowTable(2, 4))
         local paral = nn.ParallelTable()
         local seq = nn.Sequential()
         seq:add(nn.Contiguous())
         seq:add(nn.View(1, 288, 816):setNumInputDims(2))
         local conv = cudnn.SpatialConvolution(1, 1, 9, 9, 1, 1, 4, 4)
         conv.weight:fill(1/81)
         conv.bias:fill(0)
         seq:add(conv)
         paral:add(seq)
         for i=1, 3 do
            paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
         end
         avg:add(paral)
         avg:add(nn.JoinTable(1, 3))
         model:add(nn.Sequencer(avg:cuda()))
      end
      model:cuda()
   elseif opt.rnnPath ~= 'none' then                 -- For testing end2end trained CNN-RNN
      local Rnn
      assert(paths.filep(opt.rnnPath), 'Pretrained RNN not found: ' .. opt.rnnPath)
      print('=> Loading pretrained RNN from ' .. opt.rnnPath)
      Rnn = torch.load(opt.rnnPath)
      print(Rnn)
      model = nn.Sequential()
      model:add(Rnn)
      model:add(nn.Sequencer(cudnn.SpatialSoftMax()))
      model:cuda()
   else                                                             -- Initialize RNN
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end


   --- Enet encoder and decoder ---

   --[[part_1 = model:get(27):get(1):clone()
   part_2 = model:get(27):get(2):clone()
   part_1:remove(6)
   part_1:add(nn.SpatialFullConvolution(16, 5, 2, 2, 2, 2))
   part_2:get(6):remove(6)
   part_2:get(6):insert(nn.Linear(128, 4), 6)
   part_2:get(6):remove(4)
   part_2:get(6):remove(3)
   part_2:get(6):insert(nn.View(4600), 3)
   part_2:get(6):insert(nn.Linear(4600, 128), 4)

   concat_1 = nn.ConcatTable()
   concat_1:add(part_1)
   concat_1:add(part_2)
   model:remove(27)
   model:add(concat_1)]]--

   --------------------------------

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            --local rnn = require 'rnn'
            --local ConvLSTM = require 'ConvLSTM'
            --local HoleConvLSTM = require 'HoleConvLSTM'
            --local ConvGRU = require 'ConvGRU'
            --local PixelSplit = require 'PixelSplit'
            --local PixelMerge = require 'PixelMerge'
            --local TensorSplit = require 'TensorSplit'
            --local TensorMerge = require 'TensorMerge'
            --local PixelDeshuffle = require 'PixelDeshuffle'
            local ParallelCriterion2 = require 'ParallelCriterion2'
            --local SpatialConvolutionH = require 'SpatialConvolutionH'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   
   local weights = torch.Tensor(opt.nLane+1)
   weights[1] = opt.backWeight
   for i=1,opt.nLane do
      weights[i+1] = 1
   end
   local criterion
   if (opt.netType == 'lrcn' or opt.netType == 'convlstm' or opt.netType == 'holeconvlstm' or opt.netType == 'convlstm_2c' or opt.netType == 'convgru' or opt.netType == 'convrnn' or opt.netType == 'convrnn2' or opt.netType == 'convext') and opt.lastLoss == false then
      criterion = nn.SequencerCriterion(cudnn.SpatialCrossEntropyCriterion(weights):cuda())
   elseif opt.dataset == 'coord' then
      criterion = nn.SmoothL1Criterion():cuda()
   else
      if opt.dataset == 'imgSeg' then
         print('dataset: imgSeg')
         local weight = torch.Tensor(20)
         weight:fill(1)
         weight[20] = 0
         criterion = cudnn.SpatialCrossEntropyCriterion(weight):cuda()
      elseif opt.labelType == 'seg' then
         criterion = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
      elseif opt.labelType == 'exist' then
         criterion = nn.BCECriterion():cuda()
      elseif opt.labelType == 'reg' then
         local BCE = nn.BCECriterion():cuda()
         local SL1 = nn.SmoothL1Criterion():cuda()
         criterion = nn.ParallelCriterion2():add(BCE, 0.1):add(SL1, 10)
      elseif opt.labelType == 'segExist' then
         local SCE = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
         local BCE = nn.BCECriterion():cuda()
         criterion = nn.ParallelCriterion2():add(SCE, 1):add(BCE, 0.1)
      elseif opt.labelType == 'segReg' then
         local SCE = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
         local BCE = nn.BCECriterion():cuda()
         local SL1 = nn.SmoothL1Criterion():cuda()
         criterion = nn.ParallelCriterion2():add(SCE, 0.02):add(BCE, 0.1):add(SL1, 10)
      else
         cmd:error('unknown labelType: ' .. opt.labelType)
      end
   end
   print('Model:\n' .. model:__tostring())
   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

return M
