import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import math

def show(raw_file, pred_lanes):
    img = cv2.imread(raw_file)
    y_samples = range(160,720,10)
    pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
    img_vis = img.copy()

    for lane in pred_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0,0,255), thickness=2)

    plt.imshow(img_vis)
    plt.show()

def isShort(lane, thr):
    start = [ i for i, x in enumerate(lane) if x>0 ]
    if not start:
        return 1
    start = start[0]
    end = [ i for i, x in reversed(list(enumerate(lane))) if x>0 ][0]
    x_s = lane[start]
    x_e = lane[end]
    length = math.sqrt((x_s-x_e)*(x_s-x_e)+(start-end)*(start-end)*100)
    if length <= thr:
        return 1
    else:
        return 0

def rmShort(lanes, thr):
    Lanes = [lane for lane in lanes if not isShort(lane, thr)]
    return Lanes

def connect(lanes):
    Lanes = []
    isAdd = [0 for i in range(len(lanes))]
    for i in range(len(lanes)-1):
        for j in range(i+1, len(lanes)):
            lanea = lanes[i]
            laneb = lanes[j]
            starta = [ k for k, x in enumerate(lanea) if x>0 ][0]
            startb = [ k for k, x in enumerate(laneb) if x>0 ][0]
            enda = [ k for k, x in reversed(list(enumerate(lanea))) if x>0 ][0]
            endb = [ k for k, x in reversed(list(enumerate(laneb))) if x>0 ][0]
            if enda > startb or endb > starta:
                if enda < startb:
                    lane1 = lanea[:]
                    lane2 = laneb[:]
                    start1 = starta
                    start2 = startb
                    end1 = enda
                    end2 = endb
                else:
                    lane1 = laneb[:]
                    lane2 = lanea[:]
                    start1 = startb
                    start2 = starta
                    end1 = endb
                    end2 = enda
                id1 = max(start1, end1-5)
                k1 = float(lane1[end1]-lane1[id1])/(10*(end1-id1))
                id2 = min(end2, start2+5)
                k2 = float(lane2[id2]-lane2[start2])/(10*(id2-start2))
                extend = lane1[end1] + k1 * (start2 - end1) * 10
                if abs(k1-k2) < 0.2 and abs(extend-lane2[start2]) < 50:
                    newlane = [-2 for k in range(56)]
                    newlane[start1:end1+1] = lane1[start1:end1+1]
                    newlane[start2:end2+1] = lane2[start2:end2+1]
                    Lanes.append(newlane)
                    isAdd[i] = 1
                    isAdd[j] = 1
    for i, lane in enumerate(lanes):
        if isAdd[i] == 0:
            Lanes.append(lane)
    for lane in Lanes:
        lane = fixGap(lane)
    return Lanes

def cutMax(lanes):
    start = []
    for lane in lanes:
        s = [ k for k, x in enumerate(lane) if x>0 ][0]
        start.append(s)
    m = min(start)
    if m < 350:
        index1 = start.index(m)
        reverse = start[:]
        reverse.reverse()
        index2 = reverse.index(m)
        if index1 + index2 == len(start) - 1:
            lanes[index1][m] = -2
    return lanes

def maxArea(probs, h, w):
    Probs = []
    binary = np.zeros((h, w), np.uint8)
    binary[probs > 40] = 1   # 40 60
    # print(sum(sum(binary)))
    contours0, hierarchy = cv2.findContours( binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # _, contours0, hierarchy
    areas = [cv2.contourArea(contour) for contour in contours0]
    if not areas:
        return [probs]
    #print(areas)
    max_area = max(areas)
    max_id = [i for i, area in enumerate(areas) if area == max_area][0]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours( mask, contours0, max_id, 1, -1, 8, hierarchy, 0)
    probs1 = np.multiply(probs, mask)
    Probs.append(probs1)
    if len(areas) > 1:
        areas2 = areas[:]
        areas2.remove(max_area)
        max_area2 = max(areas2)
        if max_area2 > 800:  # 800 1500
            max_id2 = [i for i, area in enumerate(areas) if area == max_area2][0]
            mask2 = np.zeros((h, w), np.uint8)
            cv2.drawContours( mask2, contours0, max_id2, 1, -1, 8, hierarchy, 0)
            probs2 = np.multiply(probs, mask2)
            Probs.append(probs2)
    return Probs

def fixGap(coordinate):
    if any(x > 0 for x in coordinate):
        start = [ i for i, x in enumerate(coordinate) if x>0 ][0]
        end = [ i for i, x in reversed(list(enumerate(coordinate))) if x>0 ][0]
        lane = coordinate[start:end+1]
        if any(x < 0 for x in lane):
            print('gap detected!')
            gap_start = [ i for i, x in enumerate(lane[:-1]) if x>0 and lane[i+1]<0 ]
            gap_end = [ i+1 for i, x in enumerate(lane[:-1]) if x<0 and lane[i+1]>0 ]
            gap_id = [ i for i, x in enumerate(lane) if x<0 ]
            for id in gap_id:
                for i in range(len(gap_start)):
                    if id > gap_start[i] and id < gap_end[i]:
                        gap_width = float(gap_end[i] - gap_start[i])
                        lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (gap_end[i] - id) / gap_width * lane[gap_start[i]])
            assert all(x > 0 for x in lane), "Gaps still exist!"
            coordinate[start:end+1] = lane
    return coordinate

def getLane(probs, thr, h, w):
    Coordinate = []
    Probs = maxArea(probs, h, w)
    for probs in Probs:
        coordinate = [-2 for i in range(56)]
        probs = cv2.resize(probs, (1280, 720), interpolation = cv2.INTER_LINEAR)
        for i, y in enumerate(range(160, 720, 10)):
            row = probs[y, :]
            m = max(row)
            x = row.argmax()
            if float(m)/255 > thr:
                coordinate[i] = x
        coordinate = fixGap(coordinate)
        Coordinate.append(coordinate)
    '''start = [ i for i, y in enumerate(coordinate) if y>0]
    if start:
        start = start[0]
        if start <= 10:
            coordinate[start] = -2'''
    return Coordinate

test = 'test'
h = 368 # 720 # 368
w = 640 # 1280 # 640
#exp = 'r101_SCNN_w8_6_ft_all'
if test == 'val':
    exp = 'r50_6_ft'
    List = open('list/val.txt', 'r')
    pred = open('pred_val_r50_6_03_03.json', 'w')
else:
    exp = 'predicts/ENet_new/r101_SCNN_w8_6_full_all_ft'
    #exp = 'r101_DUC_ft'
    List = open('list/list_test.txt', 'r')
    pred = open('pred_ENet_test.json', 'w')

thr = 0.2
Lines = List.readlines()

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

for n in range(0,2782):
    print(n)
    line = Lines[n]
    img_path = line.split()[0]
    exist_path = exp + img_path[:-3] + 'exist.txt'
    exist = open(exist_path, 'r').readline().split()
    time = int(float(exist[6])*1000)
    exist = [int(e) for e in exist[:6]]
    lanes = []
    for i in range(len(exist)):
        if exist[i] == 1:
            prob_path = exp + img_path[:-4] + '_' + str(i+1) + '_avg.png'
            probs = cv2.imread(prob_path, 0)
            Coordinate = getLane(probs, thr, h, w)
            for coordinate in Coordinate:
                lanes.append(coordinate)
    lanes = rmShort(lanes, 20)
    lanes = connect(lanes)
    lanes = rmShort(lanes, 70)
    lanes = cutMax(lanes)
    data = {'raw_file' : img_path[1:], 'lanes' : lanes, 'run_time' : time}
    js = json.dumps(data, default=default)
    #show(img_path[1:], lanes)
    pred.write(js)
    pred.write('\n')
