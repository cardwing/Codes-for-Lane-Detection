from PIL import Image,ImageDraw, ImageFilter
from erf_settings import *
import numpy as np

# Alternative to matlab script that converts probability maps to lines

def GetLane(score, thr = 0.3):

    coordinate = np.zeros(POINTS_COUNT)
    for i in range (POINTS_COUNT):
        lineId = int(TRAIN_IMG_H - i * 20 / IN_IMAGE_H_AFTER_CROP * TRAIN_IMG_H - 1)
        line = score[lineId, :]
        max_id = np.argmax(line)
        max_values = line[max_id]
        if max_values / 255.0 > thr:
            coordinate[i] = max_id

    coordSum = np.sum(coordinate > 0)
    if coordSum < 2:
        coordinate = np.zeros(POINTS_COUNT)

    return coordinate, coordSum


def GetLines(existArray, scoreMaps, thr = 0.3):
    coordinates = []

    for l in range(len(scoreMaps)):
        if (existArray[l] or l == 0):
            coordinate, coordSum = GetLane(scoreMaps[l], thr)

            if (coordSum > 1):
                xs = coordinate * (IN_IMAGE_W / TRAIN_IMG_W)
                xs = np.round(xs).astype(np.int)
                pos = xs > 0
                curY = YS[pos]
                curX = xs[pos]
                curX += 1
                coordinates.append(list(zip(curX, curY)))
            else:
                coordinates.append([])
        else:
            coordinates.append([])

    return coordinates

def AddMask(img, mask, color, threshold = 0.3):
    back = Image.new('RGB', (img.size[0], img.size[1]), color=color)

    alpha = np.array(mask).astype(float) / 255
    alpha[alpha > threshold] = 1.0
    alpha[alpha <= threshold] = 0.0
    alpha *= 255
    alpha = alpha.astype(np.uint8)
    mask = Image.fromarray(np.array(alpha), 'L')
    mask_blur = mask.filter(ImageFilter.GaussianBlur(3))

    res = Image.composite(back, img, mask_blur)
    return res
