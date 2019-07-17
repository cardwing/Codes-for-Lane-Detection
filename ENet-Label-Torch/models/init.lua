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
   elseif opt.retrain ~= 'none' then                                -- For fine tuning CNN
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
      model.__memoryOptimized = nil
   elseif opt.model ~= 'none' then                                 -- For testing CNN
      assert(paths.filep(opt.model), 'File not found: ' .. opt.model)   
      print('Loading model from file: ' .. opt.model)
      model = torch.load(opt.model):cuda()
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   --[[concat_table_1 = nn.ConcatTable()
   concat_table_2 = nn.ConcatTable()
   seq_table_1 = nn.Sequential()

   seq_2 = nn.Sequential()
   seq_3 = nn.Sequential()

   seq_2:add(nn.Power(2))
   seq_2:add(nn.Sum(1, 3))
   seq_2:add(nn.SpatialUpSamplingBilinear(8))
   seq_2:add(nn.SpatialSoftMax())


   seq_3:add(nn.Power(2))
   seq_3:add(nn.Sum(1, 3))
   seq_3:add(nn.SpatialUpSamplingBilinear(8))
   seq_3:add(nn.SpatialSoftMax())

   concat_table_1:add(model:get(22):clone())
   concat_table_1:add(seq_2)
   model:remove(22)
   model:add(concat_table_1)

   for i = 19, 22 do 
      seq_table_1:add(model:get(i):clone())
   end   

   for i = 19, 22 do
      model:remove(19)
   end

   concat_table_2:add(seq_table_1)
   concat_table_2:add(seq_3)

   model:add(concat_table_2)
   model:add(nn.FlattenTable())]]--

   model:get(19):get(1):get(4):get(1):get(2):get(6):remove(3)
   model:get(19):get(1):get(4):get(1):get(2):get(6):insert(nn.View(3965), 3)
   model:get(19):get(1):get(4):get(1):get(2):get(6):remove(4)
   model:get(19):get(1):get(4):get(1):get(2):get(6):insert(nn.Linear(3965, 128), 4)


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
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   
   local weights = torch.Tensor(5)
   weights[1] = opt.backWeight
   weights[2] = 1
   weights[3] = 1
   weights[4] = 1
   weights[5] = 1
   local criterion
   if opt.dataset == 'lane' then
      local SCE = cudnn.SpatialCrossEntropyCriterion(weights):cuda()
      local BCE = nn.BCECriterion():cuda()
      local MSE_1 = nn.MSECriterion():cuda()
      local MSE_2 = nn.MSECriterion():cuda()
      criterion = nn.ParallelCriterion2():add(SCE, 1):add(BCE, 0.1):add(MSE_1, 0.0):add(MSE_2, 0.0) -- set the coefficients of MSE_1 and MSE_2 to be 0.1 if you want to use the distillation loss
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
