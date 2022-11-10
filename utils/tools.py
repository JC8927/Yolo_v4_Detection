# coding:utf-8
# tools

import os
from os import path
from pyexpat.errors import codes
import time
from unittest import result
import cv2
import random
from xml.dom.minidom import parse

import numpy as np
import pyzbar.pyzbar as pyzbar

'''
##################### about file #####################
'''
##################-barcode圖像前處理function-################
def barcode(gray):
    texts = pyzbar.decode(gray)
    if texts == []:
        angle = barcode_angle(gray)
        if angle < -45:
            angle = -90 - angle
        texts = bar(gray, angle)
    if texts == []:
        gray = np.uint8(np.clip((1.1 * gray + 10), 0, 255))
        angle = barcode_angle(gray)
        if angle < -45:
            angle = -90 - angle
        texts = bar(gray, angle)
    return texts

def bar(image, angle):
    gray = image
    bar = rotate_bound(gray, 0 - angle)
    roi = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
    texts = pyzbar.decode(roi)
    return texts

def barcode_angle(image):
    gray = image
    ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion = cv2.erode(erosion, kernel, iterations=1)
    erosion = cv2.erode(erosion, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        rect = [0, 0, 0]
    else:
        rect = cv2.minAreaRect(contours[0])
    return rect[2]

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

# read file content
def read_file(file_name):
    '''
    read all content in file_name
    return: list 
    '''
    if not path.isfile(file_name):
        return None
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            result.append(line)
    return result


# write file
def write_file(file_name, line, write_time=False):
    '''
    file_name: name
    line: content to write
    write_time: write current time before this line
    '''
    with open(file_name, 'a') as f:
        if write_time:
            line = get_curr_date() + '\n' + str(line)
        f.write(str(line) + '\n')
    return None


# rewrite a list to file_name
def rewrite_file(file_name, ls_line):
    '''
    rewrite file in file_name
    '''
    with open(file_name, 'w') as f:
        for line in ls_line:
            f.write(str(line) + '\n')
    return


# parameter voc xml file
def parse_voc_xml(file_name, names_dict):
    '''
    return [ [id1, x1, y1, w1, h1], [id2, x2, y2, w2, h2], ... ]
    '''
    # print(file_name)
    # print(names_dict)
    result = []
    if not os.path.isfile(file_name):
        return None
    doc = parse(file_name)
    root = doc.documentElement
    size = root.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)

    objs = root.getElementsByTagName('object')
    for obj in objs:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        name_id = names_dict[name]

        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        x = (xmax + xmin) / 2.0 / width
        w = (xmax - xmin) / width
        y = (ymax + ymin) / 2.0 / height
        h = (ymax - ymin) / height

        result.append([name_id, x, y, w, h])
    return result


'''
######################## about time ####################
'''


# get current time
def get_curr_date():
    '''
    return : year-month-day-hours-minute-second
    '''
    t = time.gmtime()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", t)
    return time_str


'''
######################## about image ####################
'''


# read image
def read_img(file_name):
    '''
    read image as BGR
    return:BGR image
    '''
    if not path.exists(file_name):
        return None
    img = cv2.imread(file_name)
    return img


# draw some box on image
def draw_img(img, boxes, score, label, word_dict, color_table, ):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4], x_min, y_min, x_max, y_max
    score:[V], score of corresponding box 
    label:[V], label of corresponding box
    word_dict: dictionary of  id=>name
    return : a image after draw the boxes
    '''
    w = img.shape[1]
    h = img.shape[0]
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    ori_decode_output = []
    crop_decode_output=[]
    # 將每個box的位置資料存到yolo_box.txt中
    # 用crop_coordinates存放要剪掉的code座標
    crop_coordinates = []
    img_decode=img.copy()#保存原始圖
    for i in range(len(boxes)):
        boxes[i][0] = constrait(boxes[i][0], 0, 1)
        boxes[i][1] = constrait(boxes[i][1], 0, 1)
        boxes[i][2] = constrait(boxes[i][2], 0, 1)
        boxes[i][3] = constrait(boxes[i][3], 0, 1)
        x_min, x_max = int(boxes[i][0] * w), int(boxes[i][2] * w)
        y_min, y_max = int(boxes[i][1] * h), int(boxes[i][3] * h)

        # 將座標資訊放入crop_coordinates
        crop_coordinate = [x_min, x_max, y_min, y_max]
        crop_coordinates.append(crop_coordinate)

        curr_label = label[i] if label is not None else 0
        curr_color = color_table[curr_label] if color_table is not None else (0, 125, 255)

        ###########################################################
        # 測試裁切圖片
        #說明：對yolo_v4所偵測到的每個
        # padding設定(相對)
        # padding_X = int(0.2*(x_max-x_min))
        # padding_Y = int(0.1*(y_max-y_min))
        # padding設定(絕對)
        padding_X = 20
        padding_Y = 15

        crop_Y_min = y_min
        if crop_Y_min - padding_Y > 0:
            crop_Y_min -= padding_Y
        crop_Y_max = y_max
        if crop_Y_max + padding_Y < h:
            crop_Y_max += padding_Y
        crop_X_min = x_min
        if crop_X_min - padding_X > 0:
            crop_X_min -= padding_X
        crop_X_max = x_max
        if crop_X_max + padding_X < w:
            crop_X_max += padding_X
        crop_img = img[crop_Y_min:crop_Y_max, crop_X_min:crop_X_max]


        #用於檢查crop_img
        #cv2.namedWindow("pic",0)
        #cv2.resizeWindow("pic",crop_X_max-crop_X_min,crop_Y_max-crop_Y_min)
        #cv2.imshow("pic",crop_img)
        #cv2.imwrite('./result_dir/new_save/'+txt_name[img_num], yolo_img)
        #cv2.waitKey(0)



        # 圖片銳化函式
        def sharpen(img, sigma=50):
            # sigma = 5、15、25
            blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
            usm = cv2.addWeighted(img, 1.5, blur_img, -0.8, 0)  # 以原圖 : 模糊圖片= 1.5 : -0.5 的比例進行混合。
            return usm

        ##########################-decode-#########################
        # 對barcode進行轉正處理
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(crop_img)
        if curr_label == "barcode":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #sharpen_img = sharpen(gray, 20)  # sigma值設太大會導致偵數大幅下降
            #decoded_str = barcode(img)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #sharpen_img = sharpen(gray, 10)
            #decoded_str = pyzbar.decode(img)
        ###########################################################
        # 直接覆寫前面的curr_color(從隨機變成固定)
        if curr_label == label[0]:
            curr_color = (0, 0, 205)  # barcode 紅色
        else:
            curr_color = (215, 0, 0)  # QRcode 藍色
        ###########################################################
        # draw box
        #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), curr_color)

        # 顯示padding範圍
        cv2.rectangle(img, (crop_X_min, crop_Y_min), (crop_X_max, crop_Y_max), (144, 112, 128))

        # draw font
        draw_x_min=x_min
        draw_y_min=y_min
        for code in codes:
            data=code.data.decode('utf-8')
            # print('數據:',data)
            crop_decode_output.append(data)
            cv2.putText(img, data, (draw_x_min, draw_y_min+ 50), font, 1, curr_color)
            draw_y_min=draw_y_min+75
        if word_dict is not None:
            text_name = "{}".format(word_dict[curr_label])
            #cv2.putText(img, text_name, (x_min, y_min + 25), font, 1, curr_color)
        if score is not None:
            text_score = "{:2d}%".format(int(score[i] * 100))
            #cv2.putText(img, text_score, (x_min, y_min), font, 1, curr_color)
        # if decoded_str != [] and decoded_str[0].data.decode("utf-8") != "X":
        #     text_decoded_str = decoded_str[0].data.decode("utf-8")
        #     cv2.putText(img, text_decoded_str, (x_min, y_min+ 50), font, 1, curr_color)
        #     cv2.putText(img, text_score + " " + text_decoded_str, (x_min, y_min), font, 1, curr_color)
        #     decode_output.append(text_decoded_str)
        # else:
        #     cv2.putText(img, text_score, (x_min, y_min), font, 1, curr_color)
    img_decode = cv2.cvtColor(img_decode, cv2.COLOR_BGR2GRAY)
    codes = pyzbar.decode(img_decode)
    result_decode_output=crop_decode_output.copy()
    for code in codes:
        data=code.data.decode('utf-8')
        # print('數據:',data)
        ori_decode_output.append(data)
        pts_rect=np.array(code.rect,np.int32)
        draw_flag=True
        #檢測是否有在crop_img中偵測到此barcode/qrcode
        for i in range(len(crop_decode_output)):
            if data==crop_decode_output[i]:
                draw_flag=False
                break
        if draw_flag:
            result_decode_output.append(data)
            cv2.rectangle(img,(pts_rect[0],pts_rect[1]),(pts_rect[0]+pts_rect[2],pts_rect[1]+pts_rect[3]),(255,0,0))
            cv2.putText(img,data,(pts_rect[0],pts_rect[1]),font,1,(0,0,205))
        
    # 將每個box的位置資料存到yolo_box.txt中
    path = r'result_dir\yolo_box.txt'
    with open(path, 'w') as f:
        for crop_coordinate in crop_coordinates:
            f.write(str(crop_coordinate[0]))
            f.write(",")
            f.write(str(crop_coordinate[1]))
            f.write(",")
            f.write(str(crop_coordinate[2]))
            f.write(",")
            f.write(str(crop_coordinate[3]))
            f.write("\n")

    return img, result_decode_output


'''
######################## others ####################
'''


def get_word_dict(name_file):
    '''
    dictionary of id to name
    return:{}
    '''
    word_dict = dict()
    if not os.path.exists(name_file):
        print("Name file:{} doesn't exist".format(name_file))
    else:
        contents = read_file(name_file)
        for i in range(len(contents)):
            word_dict[i] = str(contents[i])
    return word_dict


# name => id
def word2id(names_file):
    '''
    dictionary of name to id
    return {}
    '''
    id_dict = {}
    contents = read_file(names_file)
    for i in range(len(contents)):
        id_dict[str(contents[i])] = i
    return id_dict


def constrait(x, start, end):
    '''    
    return:x    ,start <= x <= end
    '''
    if x < start:
        return start
    elif x > end:
        return end
    else:
        return x


# get a list of color of corresponding name
def get_color_table(class_num):
    '''
    return :  list of (r, g, b) color
    '''
    color_table = []
    for i in range(class_num):
        r = random.randint(128, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_table.append((b, g, r))
    return color_table


