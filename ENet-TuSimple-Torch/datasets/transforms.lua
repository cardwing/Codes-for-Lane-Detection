--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.Translate(max_x, max_y)
   return function(input, target, exist, coordinate)
      local x = torch.random(-max_x, max_x)
      local y = torch.random(-max_y/5, max_y/5)
      --print(x)
      --print(y)
      --print(exist:view(1,4))
      --print(coordinate:narrow(1,1,37):view(1,37))
      input = image.translate(input, x, torch.round(y*10*288/590))
      target:add(-1)
      target = image.translate(target, x, torch.round(y*10*288/590))
      target:add(1)
      if y > 0 then
         for i = 1, 4 do
            local start = coordinate[(i-1)*37+1]
            local over = coordinate[(i-1)*37+2]
            if start >= 0 and over >= 0 then
               if start >= y/35 then
                  coordinate[(i-1)*37+1] = start - y/35
               else
                  coordinate[(i-1)*37+1] = 0.0
               end
               if over >= (y+2)/35 then
                  coordinate[(i-1)*37+2] = over - y/35
               else
                  for j = 1, 37 do
                     coordinate[(i-1)*37+j] = -1000
                  end
                  exist[i] = 0
               end
               local startId = torch.round(coordinate[(i-1)*37+1]*35)
               local overId = torch.round(coordinate[(i-1)*37+2]*35)
               if startId >= 0 and overId >= 2 then
                  for j = startId, overId do
                     coordinate[(i-1)*37+3+j] = coordinate[(i-1)*37+3+j+y]
                  end
                  for j = overId+1, 34 do
                     coordinate[(i-1)*37+3+j] = -1000
                  end
               end
            end
         end
      elseif y < 0 then
         for i = 1, 4 do
            local start = coordinate[(i-1)*37+1]
            local over = coordinate[(i-1)*37+2]
            if start >= 0 and over >= 0 then
               coordinate[(i-1)*37+1] = start - y/35
               coordinate[(i-1)*37+2] = over - y/35
               if coordinate[(i-1)*37+2] > 34/35 then
                  coordinate[(i-1)*37+2] = 34/35
               end
               if coordinate[(i-1)*37+1] >= 33/35 then
                  for j = 1, 37 do
                     coordinate[(i-1)*37+j] = -1000
                  end
                  exist[i] = 0
               end
               local startId = torch.round(coordinate[(i-1)*37+1]*35)
               local overId = torch.round(coordinate[(i-1)*37+2]*35)
               if startId >= 0 and overId >= 2 then
                  for j = startId, overId do
                     local id = startId+overId-j
                     coordinate[(i-1)*37+3+id] = coordinate[(i-1)*37+3+id+y]
                  end
                  for j = 0, startId do
                     coordinate[(i-1)*37+3+j] = -1000
                  end
               end
            end
         end
      end
      if x > 0 then
         for i = 1, 4 do
            if exist[i] == 1 then
               local startId = torch.round(coordinate[(i-1)*37+1]*35)
               local overId = torch.round(coordinate[(i-1)*37+2]*35)
               for j = startId, overId do
                  coordinate[(i-1)*37+3+j] = coordinate[(i-1)*37+3+j] + x/816
                  if coordinate[(i-1)*37+3+j] > 1.005 then
                     coordinate[(i-1)*37+3+j] = -1000
                     if (i==3) or (i==4) then
                        coordinate[(i-1)*37+1] = coordinate[(i-1)*37+1] + 1/35
                     else
                        coordinate[(i-1)*37+2] = coordinate[(i-1)*37+2] - 1/35
                     end
                  end
               end
               if coordinate[(i-1)*37+2] - coordinate[(i-1)*37+1] <= 2/35 then
                  exist[i] = 0
                  for j = 1, 37 do
                     coordinate[(i-1)*37+j] = -1000
                  end
               end
            end
         end
      elseif x < 0 then
         for i = 1, 4 do
            if exist[i] == 1 then
               local startId = torch.round(coordinate[(i-1)*37+1]*35)
               local overId = torch.round(coordinate[(i-1)*37+2]*35)
               for j = startId, overId do
                  coordinate[(i-1)*37+3+j] = coordinate[(i-1)*37+3+j] + x/816
                  if coordinate[(i-1)*37+3+j] < -0.005 then
                     coordinate[(i-1)*37+3+j] = -1000
                     if (i==1) or (i==2) then
                        coordinate[(i-1)*37+1] = coordinate[(i-1)*37+1] + 1/35
                     else
                        coordinate[(i-1)*37+2] = coordinate[(i-1)*37+2] - 1/35
                     end
                  end
               end
               if coordinate[(i-1)*37+2] - coordinate[(i-1)*37+1] <= 2/35 then
                  exist[i] = 0
                  for j = 1, 37 do
                     coordinate[(i-1)*37+j] = -1000
                  end
               end
            end
         end
      end
      --print(exist:view(1,4))
      --print(coordinate:narrow(1,1,37):view(1,37))
      return input, target, exist, coordinate
   end
end

function M.Compose(transforms)
   return function(input, target, exist, coordinate)
      for _, transform in ipairs(transforms) do
         input, target, exist, coordinate = transform(input, target, exist, coordinate)
      end
      return input, target, exist, coordinate
   end
end

function M.ColorNormalize(meanstd)
   return function(img, target, exist, coordinate)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img, target, exist, coordinate
   end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         return image.scale(input, size, h/w * size, interpolation)
      else
         return image.scale(input, w/h * size, size, interpolation)
      end
   end
end

-- Added by PanXingang. Scales the width and height for input and target
function M.ScaleWH(w, h, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input, target, exist, coordinate)
      --print(input:size())
      --print(target:size())
      return image.scale(input, w, h, interpolation), image.scale(target, w, h, 'simple'), exist, coordinate
   end
end
	
-- Crop to centered rectangle
function M.CenterCrop(w, h)
   return function(input, target)
      local w1 = math.ceil((input:size(3) - w)/2)
      local h1 = math.ceil((input:size(2) - h)/2)
      return image.crop(input, w1, h1, w1 + w, h1 + h), image.crop(target, w1, h1, w1 + w, h1 + h) -- center patch
   end
end

function M.RandomCrop(w, h, padding)
   padding = padding or 0

   return function(input, target)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end
   
      local inputW, inputH = input:size(3), input:size(2)
      if inputW == w and inputH == h then
         return input, target
      end

      local x1, y1 = torch.random(0, inputW - w), torch.random(0, inputH - h)
      local out1 = image.crop(input, x1, y1, x1 + w, y1 + h)
      local out2 = image.crop(target, x1, y1, x1 + w, y1 + h)
      assert(out1:size(2) == h and out1:size(3) == w, 'wrong crop size')
      return out1, out2
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCropLane(w, h, padding)
   padding = padding or 0

   return function(input, target, exist, coordinate)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local inputW, inputH = input:size(3), input:size(2)
      if inputW == w and inputH == h then
         return input, target, exist, coordinate
      end
      
      target:add(-1)
      if inputH < h then
         local pad = h - inputH
         local temp = input.new(3, h, input:size(3))
         temp:zero()
            :narrow(2, pad+1, inputH)
            :copy(input)
         input = temp
         local temp2 = input.new(1, h, input:size(3))
         temp2:zero()
            :narrow(2, pad+1, inputH)
            :copy(target)
         target = temp2
      end
      if inputW < w then
         local pad = torch.random(0, w - inputW)
         local temp = input.new(3, input:size(2), w)
         temp:zero()
            :narrow(3, pad+1, inputW)
            :copy(input)
         input = temp
         local temp2 = input.new(1, input:size(2), w)
         temp2:zero()
            :narrow(3, pad+1, inputW)
            :copy(target)
         target = temp2
      end
      local inputW, inputH = input:size(3), input:size(2)
      local offsetW = inputW - w
      local x1, y1 = torch.random(torch.round(offsetW/3), torch.round(offsetW/3*2)), inputH - h
      local out1 = image.crop(input, x1, y1, x1 + w, y1 + h)
      local out2 = image.crop(target, x1, y1, x1 + w, y1 + h)
      out2:add(1)
      assert(out1:size(2) == h and out1:size(3) == w, 'wrong crop size')
      return out1, out2, exist, coordinate
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function M.TenCrop(size)
   local centerCrop = M.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
   return function(input, target)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic'), image.scale(target, targetW, targetH, 'simple')
   end
end

--Added by PanXingang. Resized with random scale and ratio
function M.RandomScaleRatio(minW, maxW, minH, maxH)
   return function(input, target, exist, coordinate)
      local w, h = input:size(3), input:size(2)

      local targetW = torch.random(minW, maxW)
      local targetH = torch.random(minH, maxH)
      
      return image.scale(input, targetW, targetH, 'bicubic'), image.scale(target, targetW, targetH, 'simple'), exist, coordinate
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size)
   local scale = M.Scale(size)
   local crop = M.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end

function M.HorizontalFlip(prob)
   return function(input, target)
      if torch.uniform() < prob then
         input = image.hflip(input)
         target = image.hflip(target)
      end
      return input, target
   end
end

function M.Rotation(deg, center)
   return function(input, target, exist, coordinate)
      if not center then
         center = 0
      end
      if deg ~= 0 then
         local u = torch.uniform()
         input = image.rotate(input, ((u - 0.5) * deg + center ) * math.pi / 180, 'bilinear')
         target:add(-1)
         target = image.rotate(target, ((u - 0.5) * deg + center ) * math.pi / 180, 'simple')
         target:add(1)
      end
      return input, target, exist, coordinate
   end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Contrast(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return input
   end
end

function M.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, M.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, M.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, M.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return M.RandomOrder(ts)
end

return M
