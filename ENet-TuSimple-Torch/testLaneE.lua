require 'torch'
require 'nn'
--require 'rnn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'paths'
require 'ffi'
--require 'TensorSplit'
--require 'TensorMerge'
image = require 'image'
local models = require 'models/init'
local opts = require 'opts'
local DataLoader = require 'dataloaderE'
local checkpoints = require 'checkpoints'

opt = opts.parse(arg)
show = false

checkpoint, optimState = checkpoints.latest(opt)
model = models.setup(opt, checkpoint)
--model = torch.load(opt.model)
--model:add(cudnn.SpatialSoftMax())
--model:cuda()
offset = 0
if opt.smooth then
   offset = 1
end

print(model)
local trainLoader, valLoader = DataLoader.create(opt)
print('data loaded')
input = torch.CudaTensor()
--target = torch.CudaTensor()
function copyInputs(sample)
   input = input or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   --target = target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
   input:resize(sample.input:size()):copy(sample.input)
   --target:resize(sample.target:size()):copy(sample.target)
   return input
end

function sleep(n)
   os.execute("sleep " .. tonumber(n))
end

function process( scoremap )
   local avg = nn.Sequential()
   avg:add(nn.SpatialSoftMax())
         avg:add(nn.SplitTable(1, 3))
         avg:add(nn.NarrowTable(2, opt.nLane))
         local paral = nn.ParallelTable()
         local seq = nn.Sequential()
         seq:add(nn.Contiguous())
         seq:add(nn.View(1, 368, 640):setNumInputDims(2))
         --seq:add(nn.View(1, 432, 768):setNumInputDims(2))
         --local conv = nn.SpatialConvolution(1, 1, 1, 1, 1, 1)
         local conv = nn.SpatialConvolution(1, 1, 5, 5, 1, 1, 2, 2)
         conv.weight:fill(1/25)
         conv.bias:fill(0)
         seq:add(conv)
         paral:add(seq)
         for i=1, opt.nLane-1 do
            paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
         end
         avg:add(paral)
         avg:add(nn.JoinTable(1, 3))
         avg:cuda()
         return avg:forward(scoremap)
end

model:evaluate()
for n, sample in valLoader:run() do
   print(n)
   local timer = torch.Timer()
   input = copyInputs(sample)
   local imgpath = sample.imgpath
   output = model:forward(input)
   local scoremap
   if opt.nLane == 1 then
      scoremap = output  --:double()
   else
      scoremap = output[1]  --:double()
   end
   if opt.smooth then
      scoremap = process(scoremap):float()
   else
      scoremap:float()
   end
   local t = timer:time().real
   print(t)
   timer:reset()
   local exist
   if opt.nLane > 1 then
      exist = output[2]:float()
   end
   local outputn
   ffi = require 'ffi'
   for b = 1, input:size(1) do
      print(ffi.string(imgpath[b]:data()))
      --local img = image.load(opt.data .. ffi.string(imgpath[b]:data()), 3, 'float')
      outputn = scoremap[{b,{},{},{}}]
      --[[line1 = outputn[{2-offset,{},{}}]
      line2 = outputn[{3-offset,{},{}}]
      line3 = outputn[{4-offset,{},{}}]
      line4 = outputn[{5-offset,{},{}}]
      lines = torch.FloatTensor(3,outputn:size(2),outputn:size(3)):fill(0)
      lines[{1,{},{}}] = line1
      lines[{2,{},{}}] = line2
      lines[{3,{},{}}] = line3
      lines[{1,{},{}}]:add(line4)
      lines[{3,{},{}}]:add(line4)]]
      --print(lines:size())
      local savePath, resPath
      local subPath = '.'
      for i = 1,opt.nLane do
         savePath = opt.save .. string.sub(ffi.string(imgpath[b]:data()), 1, -5) .. '_' .. i .. '_avg.png'
         resPath = string.sub(savePath, 1, -1)
         if i == 1 then
            j = string.find(resPath, '/')
            while j do
               subPath = subPath .. '/' .. string.sub(resPath, 1, j-1)
               if not paths.dirp(subPath) then
                  lfs.mkdir(subPath)
               end
               resPath = string.sub(resPath, j+1, -1)
               j = string.find(resPath, '/')
            end
         end
         image.save(savePath, outputn[{i+1-offset,{},{}}])
      end
      local existPath = opt.save .. string.sub(ffi.string(imgpath[b]:data()), 1, -4) .. 'exist.txt'
      local f = assert(io.open(existPath, 'w'))
      --print(exist)
      if opt.nLane > 1 then
         for i = 1,opt.nLane do
            if exist[b][i] > 0.5 then
               f:write('1 ')
            else
               f:write('0 ')
            end
         end
      end
      f:write(tostring(t))
      if show then
         --image.display(lines)
         img = sample.input[b]*0.6 + lines*0.8
         image.display(img)
         --sleep(3)
         local answer
         repeat
            io.write("Continue by entering 'n'.")
            io.flush()
            answer=io.read()
         until answer=="n"
      end
   end
end
