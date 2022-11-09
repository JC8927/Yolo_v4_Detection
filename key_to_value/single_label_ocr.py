from cmath import rect
from ctypes import resize
import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image,ImageFont
import os
import time
import math
import key_to_value
def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):

    B = brightness / 255.0
    c = contrast / 255.0  
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def img_resize(img):
    height,width=img.shape[0],img.shape[1]
    renew_length=1280#自定義最長邊愈拉長至多長
    if width/height>=1:#(width>height) 拉長width至愈調整尺寸
        img_new=cv2.resize(img,(renew_length,int(height*renew_length/width)))
    else:#(height>width) 拉長height至愈調整尺寸 
        img_new=cv2.resize(img,(int(width*renew_length/height),renew_length))
    return img_new

def sharpen(img,img_2,para_1):
    blur_img=cv2.addWeighted(img,para_1,img_2,1-para_1,0)
    return blur_img

mask_img_path="./mask_img/"#mask圖片
ori_img_path="./single_ori_img/"#原始圖片
result_img_path="./result_img/"
ocr_img_path="./ocr_img/"
noresize_ocr_img_path="./noresize_ocr_img/"

img_name=os.listdir(mask_img_path)
img_num=len(img_name)

#以mask圖片做裁剪圖片
for j in range(img_num):
    img = cv2.imread(mask_img_path+img_name[j])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV )

    contours,hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    string_bounding=[]
    crop_img_max=0

    #找出面積最大之contours
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        crop_img_size=w*h
        if crop_img_size>crop_img_max:
            crop_img_max=crop_img_size
            index=i

    cnt=contours[index]
    x,y,w,h=cv2.boundingRect(cnt)
    img=cv2.imread(ori_img_path+img_name[j])
    #cv2.imshow('original',img)
    result=img[y:y+h,x:x+w]
    #cv2.imshow('result',result)

    # # 取得紅色方框的旋轉角度 保留轉正功能 還未用
    # angle = rect[2]
    # if angle < -45:
    #     angle = 90 + angle

    # # 以影像中心為旋轉軸心
    # (h, w) = img.shape[:2]
    # center = (w // 2, h // 2)

    # # 計算旋轉矩陣
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # # 旋轉圖片
    # rotated = cv2.warpAffine(img_debug, M, (w, h),
    #         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    # img_final = cv2.warpAffine(img, M, (w, h),
    #         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow('rotated',rotated)
    # cv2.imshow('img_final',img_final)

    #cv2.waitKey(0)
    cv2.imwrite(result_img_path+img_name[j],result)

#讀取被裁減之圖片
img_path="./result_img/"
img_name=os.listdir(img_path)
img_num=len(img_name)

#進行圖片前處理
for i in range(0,img_num):
    img=cv2.imread(img_path+img_name[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mod_img=modify_contrast_and_brightness2(img,0,50)#調整圖片對比
    cv2.imwrite('./crap_mod_img/'+img_name[i],mod_img)#儲存調完對比之圖片

    ret,th1=cv2.threshold(mod_img,120,255,cv2.THRESH_BINARY)#二值化圖片
    sha_img=sharpen(mod_img,th1,0.6)#疊加二值化圖片與調完對比之圖片 0.6為兩圖佔比

    cv2.imwrite('./crap_sha_img/'+img_name[i],sha_img)#儲存疊加後之圖片

ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False) # need to run only once to download and load model into memory

img_path="./crap_sha_img/"#選擇ocr處理之圖片
img_name=os.listdir(img_path)
img_num=len(img_name)

start=time.time()
for i in range(img_num):
    

    result = ocr.ocr(result_img_path+img_name[i], cls=True)

    for line in result:
        print(line)
    print(img_name[i])
    # draw result
    font=ImageFont.load_default()
    image = Image.open(img_path+img_name[i]).convert("RGB")#開sha後圖片
    ori_img=Image.open(result_img_path+img_name[i]).convert("RGB")#開mask後圖片
    boxes = []
    txts = []
    scores = []
    imformation_list=key_to_value.data_preprocess(result)
    result_list=key_to_value.first_compare(imformation_list)
    for j in range(len(result)):
        if float(result[j][1][1])>0.8:#以分數作審核標準
            boxes.append(result[j][0])
            txts.append(result[j][1][0])
            scores.append(result[j][1][1])
    im_show = draw_ocr(ori_img, boxes, txts, scores,font_path="./simfang.ttf")
    im_show = Image.fromarray(im_show)
    im_show.save("./crap_sha_result/"+img_name[i])
end=time.time()
print("經過秒數:"+format(end-start))