require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'paths'
local ffi = require 'ffi'
image = require 'image'
local models = require 'models/init_test'
local opts = require 'opts'
local DataLoader = require 'dataloader'
local checkpoints = require 'checkpoints'

opt = opts.parse(arg)
show = false    -- Set show to true if you want to visualize. In addition, you need to use qlua instead of th.

checkpoint, optimState = checkpoints.latest(opt)
model = models.setup(opt, checkpoint)
offset = 0
if opt.smooth then
   offset = 1
end

print(model)
local valLoader = DataLoader.create(opt)
print('data loaded')
input = torch.CudaTensor()
function copyInputs(sample)
   input = input or (opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   input:resize(sample.input:size()):copy(sample.input)
   return input
end

function sleep(n)
   os.execute("sleep " .. tonumber(n))
end

function process( scoremap )
   local avg = nn.Sequential()
   avg:add(nn.SpatialSoftMax())
   avg:add(nn.SplitTable(1, 3))
   avg:add(nn.NarrowTable(2, 4))
   local paral = nn.ParallelTable()
   local seq = nn.Sequential()
   seq:add(nn.Contiguous())
   seq:add(nn.View(1, 208, 976):setNumInputDims(2))
   local conv = nn.SpatialConvolution(1, 1, 9, 9, 1, 1, 4, 4)
   conv.weight:fill(1/81)
   conv.bias:fill(0)
   seq:add(conv)
   paral:add(seq)
   for i=1, 3 do
      paral:add(seq:clone('weight', 'bias','gradWeight','gradBias'))
   end
   avg:add(paral)
   avg:add(nn.JoinTable(1, 3))
   avg:cuda()
   return avg:forward(scoremap)
end

model:evaluate()
T = 0
N = 0
for n, sample in valLoader:run() do
   print(n)
   input = copyInputs(sample)
   local imgpath = sample.imgpath
   local timer = torch.Timer()
   output = model:forward(input)
   local t = timer:time().real
   print('time: ' .. t)
   local scoremap = output[1]  --:double()
   if opt.smooth then
      scoremap = process(scoremap):float()
   else
      local softmax = nn.SpatialSoftMax():cuda()
      scoremap = softmax(scoremap):float()
   end
   if n > 1 then
      T = T + t
      N = N + 1
      print('avgtime: ' .. T/N)
   end
   timer:reset()
   local exist = output[2]:float()
   local outputn
   for b = 1, input:size(1) do
      print('img: ' .. ffi.string(imgpath[b]:data()))
      local img = image.load(opt.data .. ffi.string(imgpath[b]:data()), 3, 'float')
      outputn = scoremap[{b,{},{},{}}]
      line1 = outputn[{2-offset,{},{}}]
      line2 = outputn[{3-offset,{},{}}]
      line3 = outputn[{4-offset,{},{}}]
      line4 = outputn[{5-offset,{},{}}]
      lines = torch.FloatTensor(3,outputn:size(2),outputn:size(3)):fill(0)
      lines[{1,{},{}}] = line1
      lines[{2,{},{}}] = line2
      lines[{3,{},{}}] = line3
      lines[{1,{},{}}]:add(line4)
      lines[{3,{},{}}]:add(line4)
      local savePath, resPath
      local subPath = '.'
      for i = 1,4 do
         savePath = opt.save .. string.sub(ffi.string(imgpath[b]:data()), 1, -5) .. '_' .. i .. '_avg.png'
         resPath = savePath
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
      for i = 1,4 do
         if exist[b][i] > 0.5 then
            f:write('1 ')
         else
            f:write('0 ')
         end
      end
      if show then
         img = sample.input[b]*0.6 + lines*0.8
         image.display(img)
         local answer
         repeat
            io.write("Continue by entering 'n'.")
            io.flush()
            answer=io.read()
         until answer=="n"
      end
   end
end
