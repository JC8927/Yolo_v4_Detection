# -*- coding: UTF-8 -*-
import math
import cv2,time
import numpy
import json
import retinex
import tensorflow
import csv
#import pytesseract
import numpy as np
import io
import os
from key_to_value import ocr_result
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
import re
#import xlwt
import pyzbar.pyzbar as pyzbar
from src.YOLO import YOLO
from src.Feature_parse_tf import get_predict_result
from utils import tools
from key_to_value import key_to_value
from pathlib import Path
from dbr import *
from google.cloud import vision
from tkinter import ttk
from tkinter import *
from UI import UI
from tkinter import messagebox
from numpy import number
from PIL import Image,ImageDraw
from tkinter import *
from tkinter.messagebox import *
#from xlwt import Workbook
#from paddleocr import PaddleOCR,draw_ocr

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
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 設置GOOGLE OCR API位置
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "code-reader-5-63f024a409ed.json"

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
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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

def dbr_decode(image_path,multi_flag):
    try:
        text_results = reader.decode_file(image_path)
        # 用dbr_decode_res儲存decode結果
        dbr_decode_res = []
        if text_results != None and multi_flag == False:

            for text_result in text_results:
                barcode_text = text_result.barcode_text
                barcode_location = text_result.localization_result.localization_points
                location_mid_x=int((barcode_location[0][0]+barcode_location[1][0])/2) #找出方框x中點
                location_mid_y=int((barcode_location[1][1]+barcode_location[2][1])/2) #找出方框y中點
                diction={'text':barcode_text,'bounding_poly':barcode_location,'x':location_mid_x,'y':location_mid_y}
                dbr_decode_res.append(diction)
                # print("Barcode Text : " + text_result.barcode_text)

            dbr_decode_res=sorted(dbr_decode_res,key=lambda d:d['y'])#由y軸座標排序

        if text_results != None and multi_flag:
            for text_result in text_results:
                barcode_format=text_result.barcode_format_string
                if barcode_format =="DATAMATRIX":
                    barcode_text = text_result.barcode_text
                    barcode_text_list = barcode_text.split("|")
                    barcode_location = text_result.localization_result.localization_points
                    location_mid_x=int((barcode_location[0][0]+barcode_location[1][0])/2) #找出方框x中點
                    location_mid_y=int((barcode_location[1][1]+barcode_location[2][1])/2) #找出方框y中點
                    for text in barcode_text_list:
                        diction={'text':text,'bounding_poly':barcode_location,'x':location_mid_x,'y':location_mid_y}
                        dbr_decode_res.append(diction)
                    print("hi")
        
        return dbr_decode_res
    except BarcodeReaderError as bre:
        print(bre)
        return []

def google_detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    #用於分割的標點符號
    split_mark_list=["(" , ")" , ":"]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    document=response.full_text_annotation

    para_result_list=[]
    word_result_list=[]
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                #整理paragraph level imformation
                cur_para_result_list=[]
                cur_para_loc_list=[]
                cur_para_text=" "
                #抓paragraph level bounding box
                cur_para_loc=paragraph.bounding_box.vertices
                for loc in cur_para_loc:
                    loc_list=[loc.x,loc.y]
                    cur_para_loc_list.append(loc_list)

                for word in paragraph.words:
                    #整理word level imformation
                    cur_word_result_list=[]
                    cur_word_loc_list=[]
                    cur_word_text=""

                    #抓word level bounding box
                    cur_word_loc=word.bounding_box.vertices
                    for loc in cur_word_loc:
                        loc_list=[loc.x,loc.y]
                        cur_word_loc_list.append(loc_list)

                    mark_flag=False
                    for symbol in word.symbols:
                        for mark in split_mark_list:
                            if symbol.text==mark:
                                mark_flag=True
                                break
                        if symbol.text != "|":
                            cur_word_text=cur_word_text+symbol.text
                                        
                    if mark_flag:
                        cur_para_text=cur_para_text+" "
                        continue
                    if cur_word_text.isalnum() and cur_para_text[-1].isalnum():
                        cur_para_text=cur_para_text+" "+cur_word_text
                    else:
                        cur_para_text=cur_para_text+cur_word_text
                    cur_word_text=[cur_word_text,1]
                    cur_word_result_list.append(cur_word_loc_list)
                    cur_word_result_list.append(cur_word_text)
                    word_result_list.append(cur_word_result_list)

                print(cur_para_text.strip(" "))
                cur_para_text=[cur_para_text.strip(" "),1]
                cur_para_result_list.append(cur_para_loc_list)
                cur_para_result_list.append(cur_para_text)
                if cur_para_result_list[1][0] != '':
                    para_result_list.append(cur_para_result_list)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    # 回傳辨識結果
    return para_result_list,word_result_list




def compare(str1, str2):
    tmp1 = str1.replace(" ", "")
    tmp2 = str2.replace(" ", "")

    if tmp1 in tmp2 or tmp2 in tmp1:
        return True
    else:
        return False

def ui_generate(key_value_dict=[], exe_time=0, combined_result=[],img_path='',img_path_2=''):
    """
    input:
        key_value_dict: 與'PN', 'Date', 'QTY', 'LOT', 'COO'對應的結果。
        exe_time: 主程式執行時間。
        decode_res_list: 一維碼、二維碼執行結果。
    output:
        show UI
    """
    col_name_list=['PN', 'DATE', 'QTY', 'LOT', 'COO']
    key_value_list=[]
    col_name_value_list = []
    now_label_id = 0
    key_value_dict = sorted(key_value_dict,key=lambda d:d['label_id'])
    for col in col_name_list:
        now_label_id = 0
        col_name_value_list = []
        exist_flag=False
        for diction in key_value_dict:
            if diction['col_id'] != now_label_id:
                now_label_id = now_label_id+1
                if exist_flag == False:
                    col_name_value_list.append('')
            for key in diction.keys():
                if key == col:
                    exist_flag=True
                    col_name_value_list.append(diction.get(key))
        if len(col_name_value_list)!= now_label_id+1:
            col_name_value_list.append('')
        key_value_list.append(col_name_value_list)
    label_data_list=[]
    for i in range(len(key_value_list[0])):
        data_list=[]
        for col_value_list in key_value_list:
            data_list.append(col_value_list[i])
        label_data_list.append(data_list)

    # 輸出 OCR to CSV 結果
    # print("****** OCR to CSV 結果 *************************************")
    # print([' ', 'PN', 'DATE', 'QTY', 'LOT', 'COO'])
    # print(label_data_list)
    # print()

    # # 輸出 zbar + dbr 解碼結果
    # print("***** zbar + dbr 解碼結果 ***********************************")
    # print(combined_result)
    # print()

    # print("************************************************************")
    # print(f"執行時間: {exe_time:.4}")
    # print()
    # print()
    # print()
    window = Tk()

    # 顯示當前圖片
    img_open = Image.open(img_path)
    img_open_width, img_open_height = img_open.size

    # 如果要印出decode結果，則加長UI
    if img_open_width / img_open_height <= 1:
        height = 750
        resize_factor = 350
    else:
        height = 650
        resize_factor = 300
    if img_path_2 != '':
        resize_factor = 200
        height = 750

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
    label = Label(text="RESULT to CSV 結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
    label.pack()



    # 自動調整圖片大小

    if img_open_width / img_open_height >= 1:
        img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))

    else:
        # img_open = img_open.resize((resize_factor, int(img_open_height / img_open_width * resize_factor)))
        img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))
    img_png = ImageTk.PhotoImage(img_open)
    label_img = Label(bg='gray94', fg='blue', padx=5, pady=25, image=img_png)
    label_img.pack()
    if img_path_2!='':
        # 顯示當前圖片
        img_open_2 = Image.open(img_path_2)
        img_open_width_2, img_open_height_2 = img_open_2.size
        # 自動調整圖片大小
        if img_open_width_2 / img_open_height_2 >= 1:
            img_open_2 = img_open_2.resize((int(img_open_width_2 / img_open_height_2 * resize_factor), resize_factor))

        else:
            img_open_2 = img_open_2.resize((int(img_open_width_2 / img_open_height_2 * resize_factor), resize_factor))
            # img_open_2 = img_open_2.resize((resize_factor, int(img_open_height_2 / img_open_width_2 * resize_factor)))
        img_png_2= ImageTk.PhotoImage(img_open_2)
        label_img_2 = Label(bg='gray94', fg='blue', padx=5, pady=25, image=img_png_2).pack()

    # 設定key&value對應表格
    tree = ttk.Treeview(window, height=len(label_data_list), padding=(10, 5, 20, 20), columns=('PN', 'Date', 'QTY', 'LOT', 'COO'))
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
    for i,data_list in enumerate(label_data_list):
        tree.insert("", i, text=i, values=data_list)  # 插入資料，
    tree.pack()

    # 如果有輸入decode_res_list則印出decode結果
    # if combined_result:
    #     # 設定"解碼結果"描述
    #     label = Label(text="解碼結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
    #     label.pack()

    #     # 設定解碼結果表格
    #     text = Text(height=15, width=30, font=("Arial", 14), fg="black", state=NORMAL)

    #     # 轉換解碼結果(List2Str)
    #     decode_res = ''
    #     for result in combined_result:
    #         label_id=result['label_id']
    #         label_id="*******label_id:"+str(label_id)+"*******"
    #         decode_res += str(label_id)
    #         decode_res += '\n'
    #         barcode_result=result['barcode_result']
    #         ocr_col_result = result['col_name']+":"
    #         ocr_result=ocr_col_result+result['ocr_result']
    #         ocr_result="ocr_result:"+str(ocr_result)
    #         barcode_result="barcode_result:"+result['col_name']+":"+str(barcode_result)
    #         decode_res += str(ocr_result)
    #         decode_res += '\n'
    #         decode_res += str(barcode_result)
    #         decode_res += '\n'
    #     # 匯入解碼結果表格
    #     text.insert(END, decode_res)

    #     text.pack()

    # 顯示辨識時間
    label = Label(text=f"執行時間: {exe_time:.2} (s)", font=("Arial", 14, "bold"), padx=5, pady=25, fg="black")
    label.pack()

    Button(text='退出', command=quit_program).pack()

    window.mainloop()


def ui_generate_multi_label(key_value_list=[], exe_time=0, SN_List=[], img_path=''):
    """
    input:
        key_value_dict: 與'PN', 'SN_QTY'對應的結果。
        exe_time: 主程式執行時間。
        SN_List: 一維碼、二維碼執行結果。
    output:
        show UI
    """
    col_name_list = ['PN', 'SN_QTY']
    col_name_value_list = []
    now_label_id = 0

    window = Tk()

    # 顯示當前圖片
    img_open = Image.open(img_path)
    img_open_width, img_open_height = img_open.size

    # 如果要印出decode結果，則加長UI
    if img_open_width / img_open_height <= 1:
        height = 900
        resize_factor = 300
    else:
        height = 800
        resize_factor = 250

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
    label = Label(text="RESULT to CSV 結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
    label.pack()

    # 自動調整圖片大小

    if img_open_width / img_open_height >= 1:
        img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))

    else:
        img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))
    img_png = ImageTk.PhotoImage(img_open)
    label_img = Label(bg='gray94', fg='blue', padx=5, pady=25, image=img_png).pack()

    # 設定key&value對應表格
    tree = ttk.Treeview(window, height=1, padding=(0, 0, 0, 0), columns=('PN', 'SN_QTY'))
    tree.column("PN", width=200)
    tree.column("SN_QTY", width=100)

    tree.heading("PN", text="PN")
    tree.heading("SN_QTY", text="SN_QTY")

    # 匯入key&value辨識結果
    tree.insert("", 0, values=key_value_list)  # 插入資料，
    tree.pack()

    # 如果有輸入decode_res_list則印出decode結果
    if SN_List:
        # 設定"解碼結果"描述
        label = Label(text="解碼結果:", font=("Arial", 14, "bold"), padx=5, pady=5, fg="black")
        label.pack()

        # 設定解碼結果表格
        text = Text(height=15, width=30, font=("Arial", 14), fg="black", state=NORMAL)

        # 轉換解碼結果(List2Str)
        decode_res = ''
        for res in SN_List:
            decode_res += str(res)
            decode_res += '\n'
        # 匯入解碼結果表格
        text.insert(END, decode_res)

        text.pack()

    # 顯示辨識時間
    label = Label(text=f"執行時間: {exe_time:.2} (s)", font=("Arial", 14, "bold"), padx=5, pady=25, fg="black")
    label.pack()

    Button(text='退出', command=quit_program).pack()

    window.mainloop()

def quit_program():
    sys.exit(0)

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
    renew_length=900#自定義最長邊愈拉長至多長
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

    # 定義主功能函式
###################################### 主程式 #######################################

def real_time_obj_detection(model_path,GPU_ratio=0.8,toCSV=True,sha_crap=False,retinex=False):
    #----var
    frame_count = 0
    FPS = "0"
    label_name="test"
    frame_num = 0

    #----video streaming init
    cap, height, width, writer = video_init()

    # FILENAME = 'myvideo.avi'
    # WIDTH = 1280
    # HEIGHT = 720
    # FPS = 24.0
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(FILENAME, fourcc, FPS, (WIDTH, HEIGHT))
    #----YOLO v4 init
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)

    print("請輸入目前標籤之資料夾名稱:")
    folder_name = input()
    dir_path = "./Input_dir/real_time_img_path/"+folder_name+"/"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    decode_list = []
    while (cap.isOpened()):

            ret, img = cap.read()
            pic = numpy.array(img)
            cv2.imshow('Preview_Window', img)
            # ----按下Q鍵拍下、儲存一張照片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("Preview_Window")
                # 儲存原始照片
                image_path='./Input_dir/real_time_img_path/'+folder_name+"/"+label_name+"_"+str(frame_num)+'.jpg'
                frame_num = frame_num + 1
                cv2.imwrite(image_path, pic)
                # 儲存yolo辨識照片
                #cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)

                # ***********************************************************************
                # 從這邊開始讀取拍攝到的照片並作OCR辨識
                # ***********************************************************************

                # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
                start = time.time()
                para_ocr_result,word_ocr_result = google_detect_text(image_path)
                imformation_list=key_to_value.data_preprocess(para_ocr_result)
                config=None
                save_config_path=dir_path
                config_path=dir_path+"config.json"
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        config=json.load(f)['config']
                if config==None:
                    result_list,match_text_list=key_to_value.first_compare(imformation_list,save_config_path,image_path)
                else:
                    result_list,match_text_list=key_to_value.normal_compare(imformation_list,config,image_path)
            
                #result_list,match_text_list=ocr_result.ocr_to_result(para_ocr_result)


                # 讀取zbar解碼結果
                decode_list = []

                # dbr decode
                dbr_decode_res = dbr_decode(image_path,False)
                barcode_list = [barcode['text'] for barcode in dbr_decode_res]
                #barcode_list = key_to_value.barcode_data_preprocess()
                combined_result = key_to_value.barcode_compare_ocr(result_list,dbr_decode_res)
                key_to_value.draw_final_pic(combined_result,image_path)
                # 整合zbar與dbr decode的結果
                for dbr_result in barcode_list:
                    decode_list.append(dbr_result)

                # 印出Google OCR結果
                # print("OCR Text Part:\n")
                ocr_text=[]
                for res in para_ocr_result:
                    ocr_text.append(res[1][0])
                    # f.write(res[1][0] + '\n')
                    # print(res)

                #####################################################
                # 印出UI
                # 設定ui主畫面
                end = time.time()
                exe_time = start - end
                #ui_generate(result_list, exe_time, combined_result)

                # ----release
                decode_list = []
                
                #f.close()
                #fc.close()

            # ----按下X鍵停止錄影並結束程式
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break


    cap.release()

    cv2.destroyAllWindows()

def photo_obj_detection(model_path,GPU_ratio=0.6,toCSV=True,sha_crap=False,retinex=False,folder_path=""):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")
    mode_flag=-1
    # 資料夾裡面每個檔案
    dir_path = "./Input_dir/"
    dir_path = dir_path+folder_path+"/"
    pathlist = sorted(Path(dir_path).glob('*'))  # 用哪個資料夾裡的檔案
    #print("請選擇模式:1.單一label 2. multi label")
    #mode_flag=input()
    for path in pathlist:  # path: 每張檔案的路徑
        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        sub_name = path.name[-4:]
        if path.name[-4:]!=".jpg":
            continue
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
        print("目前照片:"+str(img_path))
        

        # 輸出googleOCR辨識結果
        result_path = './result_dir/result_txt.txt'
        decode_result_path = './result_dir/decode_result_txt.txt'

        f = open(result_path, 'w', encoding='utf-8')
        fc = open(decode_result_path, 'w', encoding='utf-8')

        para_ocr_result,word_ocr_result = google_detect_text(image_path)
        start = time.time()
        imformation_list=key_to_value.data_preprocess(para_ocr_result)
        config=None
        save_config_path=dir_path
        config_path=dir_path+"config.json"
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config=json.load(f)['config']
        if config==None:
            result_list,match_text_list=key_to_value.first_compare(imformation_list,save_config_path,image_path)
        else:
            result_list,match_text_list=key_to_value.normal_compare(imformation_list,config,image_path)
      
        #result_list,match_text_list=ocr_result.ocr_to_result(para_ocr_result)


        # 讀取zbar解碼結果
        decode_list = []

        # dbr decode
        dbr_decode_res = dbr_decode(image_path,False)
        barcode_list = [barcode['text'] for barcode in dbr_decode_res]
        #barcode_list = key_to_value.barcode_data_preprocess()
        combined_result = key_to_value.barcode_compare_ocr(result_list,dbr_decode_res)
        #key_to_value.draw_final_pic(combined_result,image_path)
        # 整合zbar與dbr decode的結果
        for dbr_result in barcode_list:
            decode_list.append(dbr_result)

        # 印出Google OCR結果
        # print("OCR Text Part:\n")
        ocr_text=[]
        for res in para_ocr_result:
            ocr_text.append(res[1][0])
            # f.write(res[1][0] + '\n')
            # print(res)

        # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n")
        for decode in decode_list:
            fc.write(decode + '\n')
            # print(decode)

        # OCR轉CSV
        if toCSV:
            toCSV_list = toCSV_processing(ocr_text)
            #print(f"toCSV_list{toCSV_list}")

        # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start

        #####################################################
        # 印出UI

        # img = cv2.imread(image_path)
        # img = img_resize(img)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # crop_y = int(1*img.shape[0])
        # crop_x = int(1*img.shape[1])
        # cv2.resizeWindow("img",crop_x,crop_y)
        # cv2.imshow("img",img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        ui_generate(result_list, exe_time, combined_result,image_path)

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


def photo_obj_detection_cloud(model_path, GPU_ratio=0.6, toCSV=True, sha_crap=False, retinex=False, folder_path=""):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")


    mode_flag = -1
    # 資料夾裡面每個檔案
    dir_path = "C:/Users/shiii/我的雲端硬碟/photo_obj_detection_cloud_folder/"
    dir_path = dir_path + folder_path + "/"
    pathlist = sorted(Path(dir_path).glob('*'))  # 用哪個資料夾裡的檔案
    # print("請選擇模式:1.單一label 2. multi label")
    # mode_flag=input()
    for path in pathlist:  # path: 每張檔案的路徑
        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        sub_name = path.name[-4:]
        if path.name[-4:] != ".jpg":
            continue
        # 讀取拍攝好的照片(result_pic_orig.jpg)
        img_path = os.path.join('.', path)
        print(img_path)
        img = mpimg.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        print("目前照片:" + str(img_path))

        # 輸出googleOCR辨識結果
        result_path = './result_dir/result_txt.txt'
        decode_result_path = './result_dir/decode_result_txt.txt'

        f = open(result_path, 'w', encoding='utf-8')
        fc = open(decode_result_path, 'w', encoding='utf-8')

        para_ocr_result, word_ocr_result = google_detect_text(image_path)
        start = time.time()
        imformation_list = key_to_value.data_preprocess(para_ocr_result)
        config = None
        save_config_path = dir_path
        config_path = dir_path + "config.json"
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)['config']
        if config == None:
            result_list, match_text_list = key_to_value.first_compare(imformation_list, save_config_path, image_path)
        else:
            result_list, match_text_list = key_to_value.normal_compare(imformation_list, config, image_path)

        # result_list,match_text_list=ocr_result.ocr_to_result(para_ocr_result)

        # 讀取zbar解碼結果
        decode_list = []

        # dbr decode
        dbr_decode_res = dbr_decode(image_path, False)
        barcode_list = [barcode['text'] for barcode in dbr_decode_res]
        # barcode_list = key_to_value.barcode_data_preprocess()
        combined_result = key_to_value.barcode_compare_ocr(result_list, dbr_decode_res)
        # key_to_value.draw_final_pic(combined_result,image_path)
        # 整合zbar與dbr decode的結果
        for dbr_result in barcode_list:
            decode_list.append(dbr_result)

        # 印出Google OCR結果
        # print("OCR Text Part:\n")
        ocr_text = []
        for res in para_ocr_result:
            ocr_text.append(res[1][0])
            # f.write(res[1][0] + '\n')
            # print(res)

        # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n")
        for decode in decode_list:
            fc.write(decode + '\n')
            # print(decode)

        # OCR轉CSV
        if toCSV:
            toCSV_list = toCSV_processing(ocr_text)
            # print(f"toCSV_list{toCSV_list}")

        # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start

        #####################################################
        # 印出UI

        # img = cv2.imread(image_path)
        # img = img_resize(img)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # crop_y = int(1*img.shape[0])
        # crop_x = int(1*img.shape[1])
        # cv2.resizeWindow("img",crop_x,crop_y)
        # cv2.imshow("img",img)
        cv2.waitKey(1)
        ui_generate(result_list, exe_time, combined_result, image_path)

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

def cross_photo_obj_detection(model_path, GPU_ratio=0.6, toCSV=True, sha_crap=False, retinex=False,folder_path=''):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    #print("yolo initial done")

    # 資料夾裡面每個檔案
    dir_path="./Input_dir/"
    dir_path_first = dir_path+folder_path+"/first/"  
    dir_path_second = dir_path+folder_path+"/second/" 
    first_img_name_list=os.listdir(dir_path_first)
    second_img_name_list=os.listdir(dir_path_second)
    first_save_config_path=dir_path_first
    second_save_config_path=dir_path_second
    pathlist_first = sorted(Path(dir_path_first).glob('*'))  # 用哪個資料夾裡的檔案
    pathlist_second = sorted(Path(dir_path_second).glob('*'))
    # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
    start = time.time()
    ocr_result = []

    for first_img_name in first_img_name_list:  # path: 每張檔案的路徑
            second_img = None
            img_exist_flag = False
            sub_name = first_img_name[-4:]
            if sub_name!=".jpg":
                continue
            # 讀取第一張照片
            first_img_path = dir_path_first+first_img_name
            first_img = cv2.imread(first_img_path)
            # 讀取第二張照片
            for second_img_name in second_img_name_list:
                if second_img_name == first_img_name:
                    second_img_path = dir_path_second+second_img_name
                    second_img = cv2.imread(second_img_path)
                    img_exist_flag = True
            if img_exist_flag == False:
                print("找不到相對應的另一面照片")
                continue

            # 輸出googleOCR辨識結果
            result_path = './result_dir/result_txt.txt'
            decode_result_path = './result_dir/decode_result_txt.txt'

            f = open(result_path, 'w', encoding='utf-8')
            fc = open(decode_result_path, 'w', encoding='utf-8')

            first_para_ocr_result,first_word_ocr_result = google_detect_text(first_img_path)
            second_para_ocr_result,second_word_ocr_result = google_detect_text(second_img_path)
            start = time.time()
            first_imformation_list=key_to_value.data_preprocess(first_para_ocr_result)
            second_imformation_list=key_to_value.data_preprocess(second_para_ocr_result)
            first_config=None
            second_config=None
            first_config_path=first_save_config_path+"config.json"
            second_config_path=second_save_config_path+"config.json"
            if os.path.isfile(first_config_path):
                with open(first_config_path) as f:
                    first_config=json.load(f)['config']
            if os.path.isfile(second_config_path):
                with open(second_config_path) as f:
                    second_config=json.load(f)['config']
            if first_config==None:
                first_result_list,first_match_text_list=key_to_value.first_compare(first_imformation_list,first_save_config_path,first_img_path)
            else:
                first_result_list,first_match_text_list=key_to_value.normal_compare(first_imformation_list,first_config,first_img_path)
            if second_config==None:
                second_result_list,second_match_text_list=key_to_value.first_compare(second_imformation_list,second_save_config_path,second_img_path)
            else:
                second_result_list,second_match_text_list=key_to_value.normal_compare(second_imformation_list,second_config,second_img_path)
        
            #result_list,match_text_list=ocr_result.ocr_to_result(para_ocr_result)


            # 讀取zbar解碼結果
            decode_list = []

            # dbr decode
            first_dbr_decode_res = dbr_decode(first_img_path,False)
            second_dbr_decode_res = dbr_decode(second_img_path,False)
            first_barcode_list = [barcode['text'] for barcode in first_dbr_decode_res]
            second_barcode_list = [barcode['text'] for barcode in second_dbr_decode_res]
            #barcode_list = key_to_value.barcode_data_preprocess()
            first_combined_result = key_to_value.barcode_compare_ocr(first_result_list,first_dbr_decode_res)
            second_combined_result = key_to_value.barcode_compare_ocr(second_result_list,second_dbr_decode_res)
            #key_to_value.draw_final_pic(first_combined_result,first_img_path)
            #key_to_value.draw_final_pic(second_combined_result,second_img_path)
            combined_result = first_combined_result+second_combined_result
            result_list = first_result_list + second_result_list
            # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
            end = time.time()

            # 用start - end算出程式運行時間，並且print出來
            exe_time = end - start

            #####################################################
            # 印出UI
            first_img = img_resize(first_img)
            second_img = img_resize(second_img)
            # cv2.namedWindow("first_img", cv2.WINDOW_NORMAL)
            # crop_y = int(1*first_img.shape[0])
            # crop_x = int(1*first_img.shape[1])
            # cv2.resizeWindow("first_img",crop_x,crop_y)
            # cv2.imshow("first_img",first_img)
            # cv2.namedWindow("second_img", cv2.WINDOW_NORMAL)
            # crop_y = int(1*second_img.shape[0])
            # crop_x = int(1*second_img.shape[1])
            # cv2.resizeWindow("second_img",crop_x,crop_y)
            # cv2.imshow("second_img",second_img)
            cv2.waitKey(1)
            ui_generate(result_list, exe_time, combined_result,first_img_path,second_img_path)

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

def multi_code_detection(model_path,GPU_ratio=0.6,toCSV=True,sha_crap=False,retinex=False,folder_path=""):
    # ----YOLO v4 init
    global os
    # yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 設定ui_generate_multi_label需要用到的參數
    key_value_list = []
    exe_time = 0
    SN_List = []
    img_path = ''


    mode_flag=-1
    # 資料夾裡面每個檔案
    dir_path = "./Input_dir/"
    dir_path = dir_path+folder_path+"/"
    pathlist = sorted(Path(dir_path).glob('*'))  # 用哪個資料夾裡的檔案
    #print("請選擇模式:1.單一label 2. multi label")
    #mode_flag=input()
    for path in pathlist:  # path: 每張檔案的路徑
        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)

        img_path = path

        sub_name = path.name[-4:]
        if path.name[-4:]!=".jpg":
            continue
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
        print("目前照片:"+str(img_path))
        

        # 輸出googleOCR辨識結果
        result_path = './result_dir/result_txt.txt'
        decode_result_path = './result_dir/decode_result_txt.txt'

        f = open(result_path, 'w', encoding='utf-8')
        fc = open(decode_result_path, 'w', encoding='utf-8')

        #para_ocr_result,word_ocr_result = google_detect_text(image_path)
        start = time.time()
        #imformation_list=key_to_value.data_preprocess(para_ocr_result)
        # config=None
        # save_config_path=dir_path
        # config_path=dir_path+"config.json"
        # if os.path.isfile(config_path):
        #     with open(config_path) as f:
        #         config=json.load(f)['config']
        # if config==None:
        #     result_list,match_text_list=key_to_value.first_compare(imformation_list,save_config_path,image_path)
        # else:
        #     result_list,match_text_list=key_to_value.normal_compare(imformation_list,config,image_path)
      
        #result_list,match_text_list=ocr_result.ocr_to_result(para_ocr_result)


        # 讀取zbar解碼結果
        decode_list = []

        # dbr decode
        multi_flag = True
        dbr_decode_res = dbr_decode(image_path,multi_flag=multi_flag)
        if dbr_decode_res!=[]:
            barcode_list = [barcode['text'] for barcode in dbr_decode_res]
            PN = barcode_list[0]
            SN = barcode_list[2:]
            print("P/N:"+PN)
            print("S/N數量:"+str(len(SN)))
            key_value_list.append(PN)
            key_value_list.append(len(SN))
            SN_List = SN
            for text in SN:
                print("S/N:"+text)
        else:
            print("偵測失敗")
            return
        #barcode_list = key_to_value.barcode_data_preprocess()
        #combined_result = key_to_value.barcode_compare_ocr(result_list,dbr_decode_res)
        #key_to_value.draw_final_pic(combined_result,image_path)
        # 整合zbar與dbr decode的結果
        # for dbr_result in barcode_list:
        #     decode_list.append(dbr_result)

        # 印出Google OCR結果
        # print("OCR Text Part:\n")
        # ocr_text=[]
        # for res in para_ocr_result:
        #     ocr_text.append(res[1][0])
            # f.write(res[1][0] + '\n')
            # print(res)

        # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n")
        # for decode in decode_list:
        #     fc.write(decode + '\n')
            # print(decode)

        # # OCR轉CSV
        # if toCSV:
        #     toCSV_list = toCSV_processing(ocr_text)
            #print(f"toCSV_list{toCSV_list}")

        # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start

        ui_generate_multi_label(key_value_list, exe_time, SN_List, img_path)
        #####################################################
        # 印出UI

        # img = cv2.imread(image_path)
        # img = img_resize(img)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # crop_y = int(1*img.shape[0])
        # crop_x = int(1*img.shape[1])
        # cv2.resizeWindow("img",crop_x,crop_y)
        # cv2.imshow("img",img)
        # cv2.waitKey(1)
        # ui_generate(result_list, exe_time, combined_result,image_path)

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
###################################### 主UI #######################################

from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


# 設計登入頁面
class LoginPage(object):
    def __init__(self, master=None):
        self.root = master  # 定義內部變數root
        self.root.geometry('%dx%d' % (400, 400))  # 設定視窗大小
        self.username = StringVar()
        self.username.set('admin')
        self.password = StringVar('')
        self.password.set('123456')
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 建立Frame
        self.page.pack()
        Label(self.page, text='Code Reader 登入系統').pack()
        Label(self.page).pack()
        Label(self.page, text='賬戶: ').pack()
        Entry(self.page, textvariable=self.username).pack()
        Label(self.page, text='密碼: ').pack()
        Entry(self.page, textvariable=self.password, show='*').pack()
        Button(self.page, text='登入', command=self.loginCheck).pack()
        Button(self.page, text='退出', command=self.quit_program).pack()

    def loginCheck(self):
        name = self.username.get()
        secret = self.password.get()
        if name == 'admin' and secret == '123456':
            self.page.destroy()
            MainPage(self.root)
        else:
            showinfo(title='錯誤', message='賬號或密碼錯誤！')

        # 設計主程式頁面

    def quit_program(self):
        sys.exit(0)

class MainPage(object):
    def __init__(self, master=None):
        self.root = master  # 定義內部變數root
        self.root.geometry('%dx%d' % (400, 400))  # 設定視窗大小
        self.createPage()

    def createPage(self):
        self.inputPage = InputFrame(self.root)  # 建立不同Frame
        self.recordPage = RecordFrame(self.root)
        self.resultPage = ResultFrame(self.root)
        self.inputPage.pack()  # 預設顯示資料錄入介面
        menubar = Menu(self.root)
        menubar.add_command(label='功能選擇', command=self.inputData)
        menubar.add_command(label='紀錄查詢', command=self.recordDisp)
        menubar.add_command(label='辨識結果', command=self.resultDisp)
        self.root['menu'] = menubar  # 設定選單欄

    def inputData(self):
        self.recordPage.pack_forget()
        self.resultPage.pack_forget()
        self.inputPage = InputFrame(self.root)  # 建立不同Frame
        self.inputPage.pack()

    def recordDisp(self):
        self.inputPage.pack_forget()
        self.resultPage.pack_forget()
        self.recordPage = RecordFrame(self.root)
        self.recordPage.pack()

    def resultDisp(self):
        self.inputPage.pack_forget()
        self.recordPage.pack_forget()
        self.resultPage = ResultFrame(self.root, root.key_value_dict, root.exe_time, root.combined_result)
        self.resultPage.pack()

    # 設計功能選擇頁面

class InputFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定義內部變數root
        self.folder_name = StringVar()                          
        self.folder_name.set('')
        self.box = ttk.Combobox(root, textvariable=self.folder_name, state='readonly', values=['siliconlab_box_1','siliconlab_box_2','siliconlab_box_3','siliconlab_box_4','skywork_box','skywork_disk','melexis_disk_1','melexis_disk_2','STM_disk','multi_code','test'])
        self.createPage()

    def createPage(self):
        Label(self, text='Code Reader 功能選擇').pack()
        Label(self).pack()
        # Label(self, text='即時錄影偵測: ').pack()
        # Button(self, text='開始偵測', command=self.real_time_obj_detection).pack()
        Label(self, text='本地相片偵測: ').pack()
        Button(self, text='開始偵測', command=self.UI_photo_obj_detection).pack()
        Label(self, text='雲端相片偵測: ').pack()
        Button(self, text='開始偵測', command=self.photo_obj_detection_cloud).pack()
        Label(self, text='跨面標籤偵測: ').pack()
        Button(self, text='開始偵測', command=self.UI_cross_photo_obj_detection).pack()
        Label(self, text='密集條碼偵測: ').pack()
        Button(self, text='開始偵測', command=self.UI_multi_code_detection).pack()
        Button(self, text='退出', command=self.quit_program).pack()

        
        # 設置comboBox讓使用者選擇要用哪個資料夾
        self.box.pack()

    def quit_program(self):
        sys.exit(0)

    def real_time_obj_detection(self):
        print('real_time_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
        #real_time_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True,folder_path=folder_name)

    def UI_photo_obj_detection(self):
        print('photo_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
        photo_obj_detection(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_path=folder_name)
        
    def photo_obj_detection_cloud(self):
        print('photo_obj_detection_cloud')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
        photo_obj_detection_cloud(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_path=folder_name)

    def UI_cross_photo_obj_detection(self):
        print('cross_photo_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
        cross_photo_obj_detection(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_path=folder_name)

    def UI_multi_code_detection(self):
        print('cross_photo_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
        multi_code_detection(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_path=folder_name)

class RecordFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定義內部變數root
        self.itemName = StringVar()
        self.createPage()

    def createPage(self):
        Label(self, text='查詢介面').pack()

class ResultFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None, key_value_dict=[], exe_time=0, combined_result=[]):
        Frame.__init__(self, master)
        self.key_value_dict = key_value_dict
        self.exe_time = exe_time
        self.combined_result = combined_result
        self.root = master  # 定義內部變數root
        self.createPage()

    def inputData(self):
        self.pack_forget()

    def createPage(self):
        # 設定變數
        col_name_list = ['PN', 'DATE', 'QTY', 'LOT', 'COO']
        key_value_list = []

        # init的時候甚麼都不做
        if self.key_value_dict == []:
            # 顯示頁面標題: "尚無辨識結果"
            label = Label(self, text="尚無辨識結果", font=("Arial", 20, "bold"), padx=5, pady=5, fg="black").pack()
        else:
            # 轉換輸入資訊
            for col in col_name_list:
                now_label_id = 0
                col_name_value_list = []
                exist_flag = False
                for diction in self.key_value_dict:
                    if diction['label_id'] != now_label_id:
                        now_label_id = now_label_id + 1
                        if exist_flag == False:
                            col_name_value_list.append('')
                    for key in diction.keys():
                        if key == col:
                            exist_flag = True
                            col_name_value_list.append(diction.get(key))
                if len(col_name_value_list) != now_label_id + 1:
                    col_name_value_list.append('')
                key_value_list.append(col_name_value_list)
            label_data_list = []

            for i in range(len(key_value_list[0])):
                data_list = []
                for col_value_list in key_value_list:
                    data_list.append(col_value_list[i])
                label_data_list.append(data_list)

            # 如果要印出decode結果，則加長UI
            #             if self.combined_result:
            #                 height = 650
            #             else:
            #                 height = 350

            # 顯示頁面標題: "Code Reader"
            label = Label(text="Code Reader", font=("Arial", 20, "bold"), padx=5, pady=5, fg="black")
            label.pack()

            # 顯示當前圖片
            img_open = Image.open(r'普通標籤.jpg')
            img_open_width, img_open_height = img_open.size
            # 自動調整圖片大小
            resize_factor = 300
            if img_open_width / img_open_height >= 1:
                img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))

            else:
                img_open = img_open.resize((resize_factor, int(img_open_height / img_open_width * resize_factor)))
            img_png = ImageTk.PhotoImage(img_open)
            label_img = Label(bg='gray94', fg='blue', padx=5, pady=25, image=img_png).pack()

            # 加入辨識結果對應表格
            tree = ttk.Treeview(root, height=len(label_data_list), padding=(10, 5, 20, 20),
                                columns=('PN', 'Date', 'QTY', 'LOT', 'COO'))
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

            for i, data_list in enumerate(label_data_list):
                tree.insert("", i, text=i, values=data_list)  # 插入資料，
            tree.pack()

        

            # 顯示辨識時間
            label = Label(text=f"執行時間: {self.exe_time:.2} (s)", font=("Arial", 14, "bold"), padx=5, pady=25, fg="black")
            label.pack()
            button = Button(text='繼續').pack()
            button = Button(text='回到主頁面', command=self.inputData).pack()
            root.mainloop()  # 執行視窗

if __name__ == "__main__":
    model_path = r".\yolov4-obj_best_416.ckpt.meta"
    GPU_ratio = 0.8

    while True:
        root = Tk()
        root.title('Code reader')
        LoginPage(root)
        root.mainloop()
    #real_time_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)
    #photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)
    #photo_obj_detection_cloud(model_path, GPU_ratio=GPU_ratio, toCSV=True)
    #cross_photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)




