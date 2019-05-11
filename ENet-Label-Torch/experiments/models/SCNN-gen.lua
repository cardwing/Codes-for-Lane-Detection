require 'nn'
require 'cudnn'

model = torch.load('vgg/vgg.t7')
last = model:get(43)

function buildPass(d,width,dim,scale,kw) --d=1:down-up d=2:right-left
   local pass = nn.Sequential()
   local length = 0
   if d==1 then
      Num = 36/scale
      length = 100/scale
   else
      Num = 100/scale
      length = 36/scale
   end
   local num = Num/width
   local function buildParal()
      local seq = nn.Sequential()
      seq:add(nn.SplitTable(d+1, 3)) --128 36
      -- view fom 100 to 1,1,100
      local paralView = nn.ParallelTable()
      for i=1,Num do
         local view = nn.Sequential()
         view:add(nn.Contiguous())
         if d==1 then
            view:add(nn.View(dim, 1, length):setNumInputDims(2))
         else
            view:add(nn.View(dim, length, 1):setNumInputDims(2))
         end
         paralView:add(view)
      end
      seq:add(paralView)
      if width > 1 then
         local concatM = nn.ConcatTable()
         for i=1,num do
            local merge = nn.Sequential()
            merge:add(nn.NarrowTable((i-1)*width+1,width))
            merge:add(nn.JoinTable(d+1,3))
            concatM:add(merge) -- 128,36,2 * 50
         end
      seq:add(concatM)
      end
      local concat = nn.ConcatTable()
      local part1 = nn.Sequential()
      part1:add(nn.SelectTable(1))
      local conv, conv2
      if d==2 then
         conv = cudnn.SpatialConvolution(dim,dim,1,kw,1,1,0,(kw-1)/2)
         conv2 = cudnn.SpatialConvolution(dim,dim,1,kw,1,1,0,(kw-1)/2)
      else
         conv = cudnn.SpatialConvolution(dim,dim,kw,1,1,1,(kw-1)/2,0)
         conv2 = cudnn.SpatialConvolution(dim,dim,kw,1,1,1,(kw-1)/2,0)
      end
      conv.bias = nil
      conv.gradBias = nil
      conv.weight:normal(0,math.sqrt(2/(kw*dim*dim*5)))
      conv2.bias = nil
      conv2.gradBias = nil
      conv2.weight:normal(0,math.sqrt(2/(kw*dim*dim*5)))
      local function buildConcat(d)
         local conc = nn.ConcatTable()
         if d == true then
            conc:add(nn.Identity())
            local seq = nn.Sequential()
            seq:add(conv:clone('weight','bias','gradWeight','gradBias'))
            seq:add(nn.ReLU(true))
            conc:add(seq)
         else
            local seq = nn.Sequential()
            seq:add(conv2:clone('weight','bias','gradWeight','gradBias'))
            seq:add(nn.ReLU(true))
            conc:add(seq)
            conc:add(nn.Identity())
         end
         return conc
      end
      part1:add(buildConcat(true))
      concat:add(part1)
      concat:add(nn.NarrowTable(2,num-1))
      seq:add(concat) -- {1, 1s}, {2, 3, ..., 18}
      seq:add(nn.FlattenTable())
      -- pass the rest 34+1 times
      for i = 1,num-1 do
         local concat = nn.ConcatTable()
         local part2 = nn.Sequential()
         part2:add(nn.NarrowTable(i+1, 2))
         part2:add(nn.CAddTable())
         if i~=num-1 then
            part2:add(buildConcat(true))
         else
            part2:add(buildConcat(false))
         end
         if i==1 then
            concat:add(nn.SelectTable(1))
         else
            concat:add(nn.NarrowTable(1, i))
         end
         concat:add(part2)
         if i==num-2 then
            concat:add(nn.SelectTable(num+1))
         elseif i~=num-1 then
            concat:add(nn.NarrowTable(i+3, num-1-i))
         end
         seq:add(concat)
         seq:add(nn.FlattenTable())
      end  -- {1, 2', 3', ..., 17'}, {18's, 18'}

      for i = 1,num-1 do
         local concat = nn.ConcatTable()
         local part2 = nn.Sequential()
         part2:add(nn.NarrowTable(num-i, 2))
         part2:add(nn.CAddTable())
         if i~=num-1 then
            part2:add(buildConcat(false))
         end
         if i==num-2 then
            concat:add(nn.SelectTable(1))
         elseif i~=num-1 then
            concat:add(nn.NarrowTable(1, num-1-i))
         end
         concat:add(part2)
         if i==1 then
            concat:add(nn.SelectTable(num+1))
         else
            concat:add(nn.NarrowTable(num+2-i, i))
         end
         seq:add(concat)
         seq:add(nn.FlattenTable())
      end  -- {1', 2'', 3'', ..., 17'', 18'}
      seq:add(nn.JoinTable(d+1,3)) --128,36,100
      return seq
   end
   pass:add(buildParal())
   return pass
end

function buildSCNN(dim, s)
   local Seq = nn.Sequential()
   Seq:add(buildPass(1,1,dim,s,9))
   Seq:add(buildPass(2,1,dim,s,9))
   return Seq
end

last:insert(buildSCNN(128, 1),7)
print(model)
torch.save('vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.t7', model)
