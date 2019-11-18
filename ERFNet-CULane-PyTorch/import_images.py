import csv
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from erf_settings import *

# Imports images from external source. In this case https://labelbox.com/docs/exporting-data/export-format-detail.
# It resizes images and converts points to new dimentions.

csvPath = r'/home/brans/Downloads/export-2019-10-09T07_33_27.212Z.csv'
baseDsDir = r'/home/brans/datasets/lanes/'
df = pd.read_csv(csvPath)


def getImgsList(path):
    valid_images = [".jpg", ".JPG", ".gif", ".png", ".tga", ".jpeg", ".JPEG"]
    imgs = []
    for ext in (valid_images):
        curpath = Path(path).glob('**/*' + ext)
        imgs.extend(curpath)

    imgs = list(map(str, imgs))
    return imgs

def add_points(rec, laneId, existance, lines, img):
    xScale = IN_IMAGE_W / img.width
    yScale = IN_IMAGE_H / img.height

    if (len(existance) > 0):
        existance += ' '

    if (laneId in rec and laneId != 'Lane_1_1'):
        existance += '1'
        points = rec[laneId][0]['geometry']

        pointsStr = ''
        kps = []
        for point in points:
            kps.append((point['x'], point['y']))

        kps = np.array(kps).astype(float)
        kps[:, 0] *= xScale
        kps[:, 1] *= yScale
        kps = np.round(kps).astype(int)

        for point in kps:
            if (len(pointsStr) > 0):
                pointsStr += ' '
            pointsStr += str(point[0]) + ' ' + str(point[1])

        lines += pointsStr + os.linesep
    else:
        existance += '0'

    return existance, lines


for ind, row in df.iterrows():
    existance = ''
    lines = ''
    filesList = ''
    dsName = str(row['Dataset Name'])
    datasetDir = os.path.join(baseDsDir, dsName)

    lable = row['Label']
    if (lable == 'Skip'):
        continue

    imageId = row['External ID']
    cPath = os.path.join('', dsName, 'data', imageId)
    fullPath = os.path.join(baseDsDir, dsName, 'data', imageId)
    img = Image.open(fullPath)

    filesList += (cPath + os.linesep)

    imageName = Path(imageId).stem
    curJson = json.loads(lable)

    img = img.resize((IN_IMAGE_H, IN_IMAGE_W), Image.BILINEAR)
    img.save(os.path.join(baseDsDir, dsName, 'data', imageId), 'PNG')

    existance, lines = add_points(curJson, 'Lane_1_1', existance, lines, img)
    existance, lines = add_points(curJson, 'Lane_2_1', existance, lines, img)
    existance, lines = add_points(curJson, 'Lane_2_2', existance, lines, img)
    existance, lines = add_points(curJson, 'Lane_3_1', existance, lines, img)

    with open(os.path.join(datasetDir, imageName + '.exist.txt'), 'a') as the_file:
        the_file.write(existance)

    with open(os.path.join(datasetDir, 'data', imageName + '.lines.txt'), 'a') as the_file:
        the_file.write(lines)

    with open(os.path.join(baseDsDir, 'filesList.txt'), 'a') as the_file:
        the_file.write(filesList)


