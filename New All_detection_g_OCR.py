# -*- coding: UTF-8 -*-
import math
import cv2,time
import numpy
import json
import retinex
import tensorflow
import csv
import pytesseract
import numpy as np
import io
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
import re
import xlwt
import pyzbar.pyzbar as pyzbar
from src.YOLO import YOLO
from src.Feature_parse_tf import get_predict_result
from utils import tools
from pathlib import Path
from dbr import *
from google.cloud import vision
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from numpy import number
from PIL import Image,ImageDraw
from xlwt import Workbook
from paddleocr import PaddleOCR,draw_ocr

################################# 檢查GPU環境 #################################
#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

################################# 設置套件環境 #################################

# 設置pytesseract API位置
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 設置GOOGLE OCR API位置
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "code-reader-4-555d8b63842d.json"

# 建立BarcodeReader
BarcodeReader.init_license("t0076oQAAADLDNLLexPCL5vfn2vtVNtjVvYQzSHAmkcuhnLZhwoyd50yzV5xlNT6PYgMhdBsXn72R4cNUcOLv82zt0jv+NFJb2RQn/4Yi6Q==")
reader = BarcodeReader()

# Barcode reader setting
settings = reader.get_runtime_settings()
settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_POSTALCODE | EnumBarcodeFormat_2.BF2_DOTCODE
settings.excepted_barcodes_count = 35
reader.update_runtime_settings(settings)

################################# 定義功能函式 #################################

def video_init(is_2_write=False,save_path=None):
    writer = None
    # cap = cv2.VideoCapture(r"http://192.168.0.133:8080/video")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#default 480
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)#default 640

    # width = 480
    # height = 640
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    '''
    ref:https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
    FourCC is a 4-byte code used to specify the video codec. 
    The list of available codes can be found in fourcc.org. 
    It is platform dependent. The following codecs work fine for me.
    In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
    In Windows: DIVX (More to be tested and added)
    In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
    FourCC code is passed as `cv.VideoWriter_fourcc('M','J','P','G')or cv.VideoWriter_fourcc(*'MJPG')` for MJPG.
    '''

    if is_2_write is True:
        #fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
        #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        fourcc = cv2.VideoWriter_fourcc(*'divx')
        if save_path is None:
            save_path = 'demo.avi'
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap,height,width,writer

def model_restore_from_pb(pb_path,node_dict,GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,#印出目前的運算是使用CPU或GPU
                                allow_soft_placement=True,#當設備不存在時允許tf選擇一个存在且可用的設備來繼續執行程式
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True  # 依照程式執行所需要的資料來自動調整
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # 手動限制GPU資源的使用
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            #----issue solution if models with batch norm
            '''
            如果是有batch normalzition，或者残差网络层，会出现：
            ValueError: Input 0 of node InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/cond_1/AssignMovingAvg/Switch was passed 
            float from InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_mean:0 incompatible with expected float_ref.
            ref:https://blog.csdn.net/dreamFlyWhere/article/details/83023256
            '''
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            tf.import_graph_def(graph_def, name='')  # 匯入計算圖

        sess.run(tf.global_variables_initializer())
        for key,value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            tf_dict[key] = node
        return sess,tf_dict

class Yolo_v4():
    def __init__(self,model_path,GPU_ratio=0.2):
        #----var
        class_num =2  # 80,36, 1
        height =416  # 416, 608
        width = 416  # 416, 608
        score_thresh = 0.7  # 0.5
        iou_thresh = 0.03  # 0.213
        max_box = 20  # 50
        anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
        anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 3, 2])
        name_file = "./barcode.names"

        node_dict = {"input": "Placeholder:0",
                     "pre_boxes": "concat_9:0",
                     "pre_score": "concat_10:0",
                     "pre_label": "concat_11:0",
                     }

        #----model extension check
        if model_path[-2:] == 'pb':
            sess, tf_dict = model_restore_from_pb(model_path, node_dict,GPU_ratio=GPU_ratio)
            tf_input = tf_dict['input']
            tf_pre_boxes = tf_dict["pre_boxes"]
            tf_pre_score = tf_dict['pre_score']
            tf_pre_label = tf_dict['pre_label']
        else:
            width = int(model_path.split("\\")[-1].split(".")[0].split("_")[-1])  # 416, 608
            height = width  # 416, 608
            yolo = YOLO()
            tf_input = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])

            feature_y1, feature_y2, feature_y3 = yolo.forward(tf_input, class_num, isTrain=False)
            tf_pre_boxes, tf_pre_score, tf_pre_label = get_predict_result(feature_y1, feature_y2, feature_y3,
                                                                 anchors[2], anchors[1], anchors[0],
                                                                 width, height, class_num,
                                                                 score_thresh=score_thresh,
                                                                 iou_thresh=iou_thresh,
                                                                 max_box=max_box)
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()
            #----GPU ratio setting
            config = tf.ConfigProto(log_device_placement=True,  # 印出目前的運算是使用CPU或GPU
                                    allow_soft_placement=True,  # 當設備不存在時允許tf選擇一个存在且可用的設備來繼續執行程式
                                    )
            if GPU_ratio is None:
                config.gpu_options.allow_growth = True  # 依照程式執行所需要的資料來自動調整
            else:
                config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # 手動限制GPU資源的使用
            sess = tf.Session(config=config)
            sess.run(init)
            saver.restore(sess, model_path[:-5])

        print("Height: {}, width: {}".format(height, width))

        #----label to class name
        label_dict = tools.get_word_dict(name_file)

        #----color of corresponding names
        color_table = tools.get_color_table(class_num)


        #----local var to global
        self.width = width
        self.height = height
        self.tf_input = tf_input
        self.pre_boxes = tf_pre_boxes
        self.pre_score = tf_pre_score
        self.pre_label = tf_pre_label
        self.sess = sess
        self.label_dict = label_dict
        self.color_table = color_table


    def detection(self,img_bgr):
        img_4d = cv2.resize(img_bgr,(self.width,self.height))
        img_4d = img_4d[:,:,::-1]
        img_4d = img_4d.astype(np.float32)
        img_4d /= 255 #255,123, 18
        img_4d = np.expand_dims(img_4d,axis=0)

        boxes, score, label = self.sess.run([self.pre_boxes, self.pre_score, self.pre_label],
                                            feed_dict={self.tf_input:img_4d})
        # test box
        # print("boxes: ",boxes)
        # print("score:",score)
        img_bgr ,decoded_str= tools.draw_img(img_bgr, boxes, score, label, self.label_dict, self.color_table)

        return img_bgr,decoded_str

def dbr_decode(image_path):
    try:
        text_results = reader.decode_file(image_path)
        # 用dbr_decode_res儲存decode結果
        dbr_decode_res = []
        if text_results != None:
            for text_result in text_results:
                dbr_decode_res.append(text_result.barcode_text)
                # print("Barcode Text : " + text_result.barcode_text)
        return dbr_decode_res
    except BarcodeReaderError as bre:
        print(bre)
        return []

def google_detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # print('Texts:')

    # 建立result_list 儲存辨識結果
    result_list = texts[0].description.split('\n')
    # result_list = str(result_list).encode("UTF-8")
    # result_list = result_list.decode("UTF-8")
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    # 回傳辨識結果
    return result_list

def compare(str1, str2):
    tmp1 = str1.replace(" ", "")
    tmp2 = str2.replace(" ", "")

    if tmp1 in tmp2 or tmp2 in tmp1:
        return True
    else:
        return False

def ui_generate(key_value_list=[], exe_time=0, decode_res_list=[]):
    """
    input:
        key_value_list: 與'PN', 'Date', 'QTY', 'LOT', 'COO'對應的結果。
        exe_time: 主程式執行時間。
        decode_res_list: 一維碼、二維碼執行結果。
    output:
        show UI
    """
    # 輸出 OCR to CSV 結果
    print("****** OCR to CSV 結果 *************************************")
    print([' ', 'PN', 'Date', 'QTY', 'LOT', 'COO'])
    print(key_value_list)
    print()

    # 輸出 zbar + dbr 解碼結果
    print("***** zbar + dbr 解碼結果 ***********************************")
    print(decode_res_list)
    print()

    print("************************************************************")
    print(f"執行時間: {exe_time:.4}")
    print()
    print()
    print()
    window = Tk()

    # 如果要印出decode結果，則加長UI
    if decode_res_list:
        height = 650
    else:
        height = 350
    screenwidth = window.winfo_screenwidth()  # 屏幕宽度
    screenheight = window.winfo_screenheight()  # 屏幕高度
    width = 1000
    x = int((screenwidth - width) / 2)
    y = int((screenheight - height) / 2)
    window.geometry(f'{width}x{height}+{x}+{y}')  # 大小以及位置

    window.title("Code Reader")
    window.minsize(width=200, height=300)
    window.config(padx=20, pady=20)
    window.resizable(width=False, height=False)
    # window.config(bg="white")

    # 設定ui名稱
    label = Label(text="Code Reader", font=("Arial", 25, "bold"), padx=5, pady=5, fg="black")
    label.pack()

    # 如果有輸入key_value_list則印出
    # 設定"OCR to CSV 結果"描述
    label = Label(text="OCR to CSV 結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
    label.pack()

    # 設定key&value對應表格
    tree = ttk.Treeview(window, height=1, padding=(10, 5, 20, 20), columns=('PN', 'Date', 'QTY', 'LOT', 'COO'))
    tree.column("PN", width=200)
    tree.column("Date", width=100)
    tree.column("QTY", width=100)
    tree.column("LOT", width=200)
    tree.column("COO", width=100)

    tree.heading("PN", text="PN")
    tree.heading("Date", text="Date")
    tree.heading("QTY", text="QTY")
    tree.heading("LOT", text="LOT")
    tree.heading("COO", text="COO")

    # 匯入key&value辨識結果
    tree.insert("", 0, text="0", values=key_value_list)  # 插入資料，
    tree.pack()

    # 如果有輸入decode_res_list則印出decode結果
    if decode_res_list:
        # 設定"解碼結果"描述
        label = Label(text="解碼結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
        label.pack()

        # 設定解碼結果表格
        text = Text(height=15, width=30, font=("Arial", 14), fg="black", state=NORMAL)

        # 轉換解碼結果(List2Str)
        decode_res = ''
        for res in decode_res_list:
            decode_res += str(res)
            decode_res += '\n'
        # 匯入解碼結果表格
        text.insert(END, decode_res)

        text.pack()

    # 顯示辨識時間
    label = Label(text=f"執行時間: {exe_time:.2} (s)", font=("Arial", 14, "bold"), padx=5, pady=25, fg="black")
    label.pack()
    #     window.after(3000, window.destroy)
    window.mainloop()

def toCSV_processing(ocr_result):

    # 標頭資訊(重要項目)
    Header = [' ', 'PN', 'Date', 'QTY', 'LOT', 'COO']

    # 設定資料IO路徑
    result_path = './result_dir/real_time_CSV/real_time.csv'

    # 檢查是否存在各公司資料夾，不存在的話就創立一個新的(包含標頭)
    if not os.path.isfile(result_path):
        with open(result_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(Header)  # 列出重要項目

    with open(result_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #     PN, Date, QTY, LOT, COO = 1, 2, 3, 4, 5  # 設定哪一項放在第幾格
        PN, Date, QTY, LOT, COO = 0, 1, 2, 3, 4  # 設定哪一項放在第幾格

        toCSV_list = ['-', '-', '-', '-', '-']
        EID = 0  # 換行用
        #     s = str(path)
        #     List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
        overwrite = [0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
        for line in ocr_result:
            line2 = line
            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割


            # PN
            if ('PN' in line2[0] or 'P/N' in line2[0]) and overwrite[PN] == 0:
                if len(line2[0]) > 1 and  len(line2)==2:
                    toCSV_list[PN] = line2[1].lstrip(' ')
                else:
                    toCSV_list[PN] = line2[0][2:].lstrip(' ')
                overwrite[PN] = 1
                EID = 0


            # Date
            elif ('Date' in line2[0] or 'DATE' in line2[0]) and overwrite[Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                else:
                    toCSV_list[Date] = line2[0][4:].lstrip(' ')
                overwrite[Date] = 1  # 填完了
                EID = 0  # 不用換行

            # QTY
            elif ('Qty' in line2[0] or r"Q'ty" in line2[0] or 'QTY'in line2[0] or 'quantity'in line2[0] or 'Quantity' in line2[0]) and overwrite[QTY] == 0:
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                else:
                    toCSV_list[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                overwrite[QTY] = 1
                EID = 0

            # LOT
            elif ('LOT' in line2[0] or 'Lot' in line2[0]) and overwrite[LOT] == 0:
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[LOT] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                else:
                    toCSV_list[LOT] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                overwrite[LOT] = 1
                EID = 0

            # COO
            elif ('COO' in line2[0] or 'Coo'in line2[0] or 'CoO'in line2[0] or 'Country' in line2[0]) and overwrite[COO] == 0:
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[COO] = line2[1].lstrip(' ')
                else:
                    toCSV_list[COO] = line2[0][3:].lstrip(' ')
                overwrite[COO] = 1
                EID = 0
            elif ('C.O.O.' in line2[0] or 'C.o.o.' in line2[0]) and overwrite[COO] == 0:
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[COO] = line2[1].lstrip(' ')
                else:
                    toCSV_list[COO] = line2[0][6:].lstrip(' ')
                overwrite[COO] = 1
                EID = 0
            elif ('MADE IN' in line2[0] or 'Made In' in line2[0]) and overwrite[COO] == 0:
                if len(line2) > 1 and  len(line2)==2:
                    toCSV_list[COO] = line2[1].lstrip(' ')
                else:
                    toCSV_list[COO] = line2[0][7:].lstrip(' ')
                overwrite[COO] = 1
                EID = 0

        #######################################
        overwrite[0] = 1

    return toCSV_list

################################### 圖像前處理函式 ###################################

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

def sha_crap_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mod_img = modify_contrast_and_brightness2(img, 0, 50)  # 調整圖片對比
    ret, th1 = cv2.threshold(mod_img, 120, 255, cv2.THRESH_BINARY)  # 二值化圖片
    sha_crap_img = sharpen(mod_img, th1, 0.6)  # 圖片銳利化
    cv2.imwrite('./result_dir/result_pic_yolo_crap_sha.jpg', sha_crap_img)
    return(sha_crap_img)

def retinex_processing(img, retinex_mode='msrcp'):
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 共有三種模式
    if retinex_mode == 'msrcr':
        # msrcr處理
        img = retinex.MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )

    if retinex_mode == 'amsrcr':
        # amsrcr處理
        img = retinex.automatedMSRCR(
            img,
            config['sigma_list']
        )

    if retinex_mode == 'msrcp':
        # msrcp處理
        img = retinex.MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']
        )
    return img

###################################### 主程式 #######################################

def real_time_obj_detection(model_path,GPU_ratio=0.8,toCSV=True,sha_crap=False,retinex=False):
    #----var
    frame_count = 0
    FPS = "0"
    d_t = 0

    #----video streaming init
    cap, height, width, writer = video_init()

    #----YOLO v4 init
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)

    decode_list = []
    while (cap.isOpened()):

        ret, img = cap.read()
        pic = numpy.array(img)

        # 建立decode_list儲存解碼內容
        decode_result_path = './result_dir/decode_result_txt.txt'

        if ret is True:
            #----YOLO v4 detection
            yolo_img,pyz_decoded_str = yolo_v4.detection(img)

            # 在錄影的過程中儲存解碼內容於decode_list
            decode_result = pyz_decoded_str
            if decode_result != []:
                for res in decode_result:
                    # 一樣的decode結果不重複紀錄
                    if res not in set(decode_list):
                        decode_list.append(res)
                        print(decode_list)
            #----FPS calculation
            if frame_count == 0:
                d_t = time.time()
            frame_count += 1
            if frame_count >= 10:
                d_t = time.time() - d_t
                FPS = "FPS=%1f" % (frame_count / d_t)
                frame_count = 0

            # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(yolo_img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3) #123, 255, 18

            #----image display
            cv2.imshow("Code Reader", yolo_img)


            # ----按下Q鍵拍下、儲存一張照片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 儲存原始照片
                cv2.imwrite('./result_dir/result_pic_orig.jpg', pic)
                # 儲存yolo辨識照片
                cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)

                # ***********************************************************************
                # 從這邊開始讀取拍攝到的照片並作OCR辨識
                # ***********************************************************************

                # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
                start = time.time()

                # 讀取拍攝好的照片(result_pic_orig.jpg)
                img_path = './result_dir/result_pic_orig.jpg'  # 用這個路徑讀取最後拍下的照片
                img = cv2.imread(img_path)

                # 做sha_crap前處理
                if sha_crap:
                    img = sha_crap_processing(img)

                # 做retinex前處理
                if retinex:
                    img = retinex_processing(img)

                # 輸出前處理後的圖片
                cv2.imwrite('./result_dir/result_pic_processing.jpg', img)



                # googleOCR辨識
                image_path = r'./result_dir/result_pic_processing.jpg'
                ocr_result = google_detect_text(image_path)

                # 輸出googleOCR辨識結果
                result_path = './result_dir/result_txt.txt'

                f = open(result_path, 'w',encoding='utf-8')
                fc = open(decode_result_path, 'w',encoding='utf-8')

                # 讀取zbar解碼結果
                decode_result = pyz_decoded_str
                # dbr decode
                dbr_decode_res = dbr_decode(image_path)

                # 整合zbar與dbr decode的結果
                for dbr_result in dbr_decode_res:
                    # 將dbr decode
                    if dbr_result not in set(decode_list):
                        decode_list.append(dbr_result)

                # 印出Google OCR結果
                # print("OCR Text Part:\n")
                for res in ocr_result:
                    f.write(res + '\n')
                    # print(res)

                # 印出Barcode/QRCode內容
                # print("Barcode/QRCode Part:\n")
                for decode in decode_list:
                    fc.write(decode+'\n')
                    # print(decode)

                # OCR轉CSV
                if toCSV:
                    toCSV_list = toCSV_processing(ocr_result)

                # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
                end = time.time()

                # 用start - end算出程式運行時間，並且print出來
                exe_time = end - start

                #####################################################
                # 印出UI
                # 設定ui主畫面
                ui_generate(toCSV_list, exe_time, decode_list)

                # ----release
                decode_list = []
                f.close()
                fc.close()

            # ----按下X鍵停止錄影並結束程式
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        else:
            print("get image failed")
            break


    yolo_v4.sess.close()
    cap.release()

    cv2.destroyAllWindows()

def photo_obj_detection(model_path,GPU_ratio=0.6,toCSV=True,sha_crap=False,retinex=False):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    pathlist = sorted(Path("./input_dir/Test_img/").glob('*'))  # 用哪個資料夾裡的檔案

    for path in pathlist:  # path: 每張檔案的路徑
        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        start = time.time()

        # 讀取拍攝好的照片(result_pic_orig.jpg)
        img_path = os.path.join('.', path)
        img = cv2.imread(img_path)

        # 做sha_crap前處理
        if sha_crap:
            img = sha_crap_processing(img)

        # 做retinex前處理
        if retinex:
            img = retinex_processing(img)

        # 輸出前處理後的圖片
        cv2.imwrite('./result_dir/result_pic_processing.jpg', img)

        # googleOCR辨識
        image_path = r'./result_dir/result_pic_processing.jpg'
        ocr_result = google_detect_text(image_path)

        # 輸出googleOCR辨識結果
        result_path = './result_dir/result_txt.txt'
        decode_result_path = './result_dir/decode_result_txt.txt'

        f = open(result_path, 'w', encoding='utf-8')
        fc = open(decode_result_path, 'w', encoding='utf-8')

        # 讀取zbar解碼結果
        decode_list = []

        # dbr decode
        dbr_decode_res = dbr_decode(image_path)

        # 整合zbar與dbr decode的結果
        for dbr_result in dbr_decode_res:
            decode_list.append(dbr_result)

        # 印出Google OCR結果
        # print("OCR Text Part:\n")
        for res in ocr_result:
            f.write(res + '\n')
            # print(res)

        # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n")
        for decode in decode_list:
            fc.write(decode + '\n')
            # print(decode)

        # OCR轉CSV
        if toCSV:
            toCSV_list = toCSV_processing(ocr_result)
            print(f"toCSV_list{toCSV_list}")

        # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start

        #####################################################
        # 印出UI
        ui_generate(toCSV_list, exe_time, decode_list)

        # ----release
        decode_list = []
        f.close()
        fc.close()


    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")

def photo_obj_detection_cloud(model_path,GPU_ratio=0.6,toCSV=True,sha_crap=False,retinex=False):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    pathlist = sorted(Path(r"C:/Users/shiii/我的雲端硬碟/code_reader_photo_detect/").glob('*'))  # 用哪個資料夾裡的檔案

    for path in pathlist:  # path: 每張檔案的路徑
        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        start = time.time()

        # 讀取拍攝好的照片(result_pic_orig.jpg)
        img_path = os.path.join('.', path)
        print(img_path)
        img = mpimg.imread(img_path)

        # 做sha_crap前處理
        if sha_crap:
            img = sha_crap_processing(img)

        # 做retinex前處理
        if retinex:
            img = retinex_processing(img)

        # 輸出前處理後的圖片
        cv2.imwrite('./result_dir/result_pic_processing.jpg', img)

        # googleOCR辨識
        image_path = r'./result_dir/result_pic_processing.jpg'
        ocr_result = google_detect_text(image_path)

        # 輸出googleOCR辨識結果
        result_path = './result_dir/result_txt.txt'
        decode_result_path = './result_dir/decode_result_txt.txt'

        f = open(result_path, 'w', encoding='utf-8')
        fc = open(decode_result_path, 'w', encoding='utf-8')

        # 讀取zbar解碼結果
        decode_list = []

        # dbr decode
        dbr_decode_res = dbr_decode(image_path)

        # 整合zbar與dbr decode的結果
        for dbr_result in dbr_decode_res:
            decode_list.append(dbr_result)

        # 印出Google OCR結果
        # print("OCR Text Part:\n")
        for res in ocr_result:
            f.write(res + '\n')
            # print(res)

        # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n")
        for decode in decode_list:
            fc.write(decode + '\n')
            # print(decode)

        # OCR轉CSV
        if toCSV:
            toCSV_list = toCSV_processing(ocr_result)
            print(f"toCSV_list{toCSV_list}")

        # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start

        #####################################################
        # 印出UI
        ui_generate(toCSV_list, exe_time, decode_list)

        # ----release
        decode_list = []
        f.close()
        fc.close()


    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")

def cross_photo_obj_detection(model_path, GPU_ratio=0.6, toCSV=True, sha_crap=False, retinex=False):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    pathlist = sorted(Path(r"C:/Users/shiii/我的雲端硬碟/cross_img_fold/").glob('*'))  # 用哪個資料夾裡的檔案

    # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
    start = time.time()
    ocr_result = []

    for path in pathlist:  # path: 每張檔案的路徑
        # 讀取拍攝好的照片(result_pic_orig.jpg)
        img_path = os.path.join('.', path)
        print(img_path)
        img = mpimg.imread(img_path)

        # 做sha_crap前處理
        if sha_crap:
            img = sha_crap_processing(img)

        # 做retinex前處理
        if retinex:
            img = retinex_processing(img)

        # 輸出前處理後的圖片
        cv2.imwrite('./result_dir/result_pic_processing.jpg', img)

        # googleOCR辨識
        image_path = r'./result_dir/result_pic_processing.jpg'
        ocr_result += google_detect_text(image_path)

    # 輸出googleOCR辨識結果
    result_path = './result_dir/result_txt.txt'
    decode_result_path = './result_dir/decode_result_txt.txt'

    f = open(result_path, 'w', encoding='utf-8')
    fc = open(decode_result_path, 'w', encoding='utf-8')

    # 讀取zbar解碼結果
    decode_list = []

    # dbr decode
    dbr_decode_res = dbr_decode(image_path)

    # 整合zbar與dbr decode的結果
    for dbr_result in dbr_decode_res:
        decode_list.append(dbr_result)

    # 印出Google OCR結果
    # print("OCR Text Part:\n")
    for res in ocr_result:
        f.write(res + '\n')
        # print(res)

    # 印出Barcode/QRCode內容
    # print("Barcode/QRCode Part:\n")
    for decode in decode_list:
        fc.write(decode + '\n')
        # print(decode)

    # OCR轉CSV
    if toCSV:
        toCSV_list = toCSV_processing(ocr_result)
        print(f"toCSV_list{toCSV_list}")

    # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
    end = time.time()

    # 用start - end算出程式運行時間，並且print出來
    exe_time = end - start

    #####################################################
    # 印出UI
    ui_generate(toCSV_list, exe_time, decode_list)

    # ----release
    decode_list = []
    f.close()
    fc.close()

    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    model_path = r".\yolov4-obj_best_416.ckpt.meta"
    GPU_ratio = 0.8
    # real_time_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)
    # photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)
    # photo_obj_detection_cloud(model_path, GPU_ratio=GPU_ratio, toCSV=True)
    cross_photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)




