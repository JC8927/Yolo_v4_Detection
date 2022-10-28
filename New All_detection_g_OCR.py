import math
import cv2,time
import numpy
import json

import retinex
from numpy import number
import tensorflow
from src.YOLO import YOLO
from src.Feature_parse_tf import get_predict_result
from utils import tools
import csv
from paddleocr import PaddleOCR,draw_ocr
import pytesseract
import os
from pathlib import Path
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import copy
import re
import xlwt
from xlwt import Workbook
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

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

#圖像前處理追加函式start#

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

#圖像前處理追加函式end#
def compare(str1, str2):
    tmp1 = str1.replace(" ", "")
    tmp2 = str2.replace(" ", "")

    if tmp1 in tmp2 or tmp2 in tmp1:
        return True
    else:
        return False


# API路徑(全域變數)
Google_json_path="C:/1Google_OCR/alien-proton-363201-9c70ccc912f8.json"

def real_time_obj_detection(model_path,GPU_ratio=0.8,toCSV=True):
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
        fc = open(decode_result_path, 'w')


        if ret is True:
            #----YOLO v4 detection
            yolo_img,pyz_decoded_str = yolo_v4.detection(img)

            # 在錄影的過程中儲存解碼內容
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
            cv2.imshow("YOLO v4 by JohnnyAI", yolo_img)

            # #----image writing
            # if writer is not None:
            #     writer.write(yolo_img)


            # ----按下Q鍵拍下、儲存一張照片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #####################################################
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
                # ----YOLO v4 variable init
                img = cv2.imread(img_path)

                # # 做sha_crap前處理
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # mod_img = modify_contrast_and_brightness2(img, 0, 50)  # 調整圖片對比
                # ret, th1 = cv2.threshold(mod_img, 120, 255, cv2.THRESH_BINARY)  # 二值化圖片
                # img = sharpen(mod_img, th1, 0.6)  # 圖片銳利化
                # cv2.imwrite('./result_dir/result_pic_yolo_crap_sha.jpg', img)

                # 做retinex前處理)
                # with open('config.json', 'r') as f:
                #     config = json.load(f)

                # 共有三種模式

                # msrcr處理
                # img = retinex.MSRCR(
                #     img,
                #     config['sigma_list'],
                #     config['G'],
                #     config['b'],
                #     config['alpha'],
                #     config['beta'],
                #     config['low_clip'],
                #     config['high_clip']
                # )

                # # amsrcr處理
                # img = retinex.automatedMSRCR(
                #     img,
                #     config['sigma_list']
                # )
                #
                # # msrcp處理
                # img = retinex.MSRCP(
                #     img,
                #     config['sigma_list'],
                #     config['low_clip'],
                #     config['high_clip']
                # )


                # 將yolo找到的條碼部分遮蔽
                # 讀取yolo找到的座標
                with open(r'.\result_dir\yolo_box.txt', 'r') as f:
                    coordinates = f.read()
                spilt_coordinates = coordinates.split("\n")

                # 遮蔽各個code的區域
                for coordinate in spilt_coordinates:
                    if len(coordinate.split(",")) > 1:
                        x_min = int(coordinate.split(",")[0])
                        x_max = int(coordinate.split(",")[1])
                        y_min = int(coordinate.split(",")[2])
                        y_max = int(coordinate.split(",")[3])

                        padding_x = 5
                        padding_y = 8

                        # x padding
                        if (x_max - x_min > 2 * padding_x):
                            x_max -= padding_x
                            x_min += padding_x

                        # y padding
                        if (y_max - y_min > 2 * padding_y):
                            y_max -= padding_y
                            y_min += padding_y

                        # 轉換x_min,x_max,y_min,y_max為x_left,y_top,w,h
                        start_point = (x_min, y_min)
                        end_point = (x_max, y_max)
                        color = (0, 0, 0)
                        # Thickness of -1 will fill the entire shape
                        thickness = -1

                        img = cv2.rectangle(img, start_point, end_point, color, thickness)
                # 儲存yolo_crop照片
                cv2.imwrite('./result_dir/result_pic_yolo_crop.jpg', img)
                print("***************************")

                decode_result = pyz_decoded_str

                # googleOCR辨識

                # 設置API位置
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Google_json_path
                image_path = r'./result_dir/result_pic_yolo_crop.jpg'

                # EntityAnnotation的說明文件
                # https: // cloud.google.com / python / docs / reference / vision / 2.2.0 / google.cloud.vision_v1.types.EntityAnnotation

                # 設定辨識function
                def google_detect_text(path):
                    """Detects text in the file."""
                    from google.cloud import vision
                    import io
                    client = vision.ImageAnnotatorClient()

                    with io.open(path, 'rb') as image_file:
                        content = image_file.read()

                    image = vision.Image(content=content)

                    response = client.text_detection(image=image)
                    texts = response.text_annotations
                    print('Texts:')

                    # 建立result_list 儲存辨識結果
                    result_list = texts[0].description.split('\n')

                    if response.error.message:
                        raise Exception(
                            '{}\nFor more info on error messages, check: '
                            'https://cloud.google.com/apis/design/errors'.format(
                                response.error.message))
                    # 回傳辨識結果
                    return result_list

                result = google_detect_text(image_path)

                # 匯出辨識結果(txt)
                result_path = './result_dir/result_txt.txt'
                f = open(result_path, 'w')
                fc = open(decode_result_path, 'w')

                # 印出PaddleOCR結果
                print("OCR Text Part:\n")
                for res in result:
                    f.write(res + '\n')
                    print(res)

                # 印出Barcode/QRCode內容
                print("Barcode/QRCode Part:\n")
                for decode in decode_list:
                    fc.write(decode+'\n')
                    print(decode)
                decode_list = []

                # OCR轉CSV
                if toCSV:
                    # 標頭資訊(重要項目)
                    Header = (' ', 'PN', 'Date', 'QTY', 'LOT', 'COO')

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

                        List = ['-', '-', '-', '-', '-']
                        EID = 0  # 換行用
                        #     s = str(path)
                        #     List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                        overwrite = [0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                        for line in result:
                            line2 = line
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割


                            # PN
                            if ('PN' in line2[0] or 'P/N' in line2[0]) and overwrite[PN] == 0:
                                if len(line2[0]) > 1 and  len(line2)==2:
                                    List[PN] = line2[1].lstrip(' ')
                                else:
                                    List[PN] = line2[0][2:].lstrip(' ')
                                overwrite[PN] = 1
                                EID = 0


                            # Date
                            elif ('Date' in line2[0] or 'DATE' in line2[0]) and overwrite[Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                                if len(line2) > 1 and  len(line2)==2:
                                    List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                                else:
                                    List[Date] = line2[0][4:].lstrip(' ')
                                overwrite[Date] = 1  # 填完了
                                EID = 0  # 不用換行

                            # QTY
                            elif ('Qty' in line2[0] or 'QTY'in line2[0] or 'quantity'in line2[0] or 'Quantity' in line2[0]) and overwrite[QTY] == 0:
                                if len(line2) > 1 and  len(line2)==2:
                                    List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                                else:
                                    List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                                overwrite[QTY] = 1
                                EID = 0

                            # LOT
                            elif ('LOT' in line2[0] or 'Lot' in line2[0]) and overwrite[LOT] == 0:
                                if len(line2) > 1 and  len(line2)==2:
                                    List[LOT] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                                else:
                                    List[LOT] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                                overwrite[LOT] = 1
                                EID = 0

                            # COO
                            elif ('COO' in line2[0] or 'Coo'in line2[0] or 'CoO'in line2[0] or 'Country' in line2[0]) and overwrite[COO] == 0:
                                if len(line2) > 1 and  len(line2)==2:
                                    List[COO] = line2[1].lstrip(' ')
                                else:
                                    List[COO] = line2[0][3:].lstrip(' ')
                                overwrite[COO] = 1
                                EID = 0
                            elif ('C.O.O.' in line2[0] or 'C.o.o.' in line2[0]) and overwrite[COO] == 0:
                                if len(line2) > 1 and  len(line2)==2:
                                    List[COO] = line2[1].lstrip(' ')
                                else:
                                    List[COO] = line2[0][6:].lstrip(' ')
                                overwrite[COO] = 1
                                EID = 0
                            elif ('MADE IN' in line2[0] or 'Made In' in line2[0]) and overwrite[COO] == 0:
                                if len(line2) > 1 and  len(line2)==2:
                                    List[COO] = line2[1].lstrip(' ')
                                else:
                                    List[COO] = line2[0][7:].lstrip(' ')
                                overwrite[COO] = 1
                                EID = 0

                        #######################################
                        overwrite[0] = 1
                        print(List)
                        if decode_result != []:
                            wrote = decode_result
                            for a in range(len(overwrite) - 2):
                                if overwrite[a] == 0:
                                    for res in range(len(decode_result)):
                                        if wrote[res] != 'wrote' and decode_result[res] != '':
                                            print(decode_result[res])
                                            List[a] = decode_result[res]
                                            overwrite[a] = 1
                                            wrote[res] = 'wrote'
                                            break
                        writer.writerow(List)  # 印出來

                # 用time的套件紀錄辨識完成的時間(用於計算程式運行時間)
                end = time.time()

                # 用start - end算出程式運行時間，並且print出來
                exe_time = end - start
                print("************************")
                print(f"執行時間: {exe_time:.4}")
                print("************************")
                #####################################################
                # ----release
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


    # if writer is not None:
    #     writer.release()
    cv2.destroyAllWindows()

def photo_obj_detection_HD(model_path,GPU_ratio=0.8,toCSV=True):

    # ----YOLO v4 init
    global os
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    pathlist = sorted(Path(r"./input_dir/HD_img/").glob('*'))  # 用哪個資料夾裡的檔案
    # pathlist = sorted(Path("./input_dir/Test_img/").glob('*'))  # 用哪個資料夾裡的檔案
    print(pathlist)

    for path in pathlist:  # path每張檔案的路徑


        img_path = os.path.join('.', path)
        print(img_path)
        # ----YOLO v4 variable init
        img = cv2.imread(img_path)

        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        start = time.time()

        # 做sha_crap前處理
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mod_img = modify_contrast_and_brightness2(img, 0, 50)  # 調整圖片對比
        # ret, th1 = cv2.threshold(mod_img, 120, 255, cv2.THRESH_BINARY)  # 二值化圖片
        # img = sharpen(mod_img, th1, 0.6)  # 疊加二值化圖片與調完對比之圖片 0.6為兩圖佔比

        # googleOCR辨識

        # 設置API位置
        import os
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Google_json_path
        image_path = r'./result_dir/result_pic_yolo_crop.jpg'

        # 設定辨識function
        def google_detect_text(path):
            """Detects text in the file."""
            from google.cloud import vision
            import io
            client = vision.ImageAnnotatorClient()

            with io.open(path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            response = client.text_detection(image=image)
            texts = response.text_annotations
            print('Texts:')

            # 建立result_list 儲存辨識結果
            result_list = texts[0].description.split('\n')

            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))
            # 回傳辨識結果
            return result_list

        result = google_detect_text(img_path)

        print("Text Part:\n")
        for res in result:
            print(res)

        # ----YOLO v4 detection-----------------
        yolo_img, pyz_decoded_str = yolo_v4.detection(img)
        decode_result = pyz_decoded_str
        # 印出Barcode/QRCode內容
        print("Barcode/QRCode Part:\n\n")
        if decode_result != []:
            for res in decode_result:
                print(res)
        else:
            print("Decode Fail")

        ####################################################

        # 儲存照片路徑
        # result_img_path = "./result_dir/"+str(path)+".jpg"
        # # 儲存yolo辨識照片
        # cv2.imwrite(result_img_path, yolo_img)
        #####################################################
        # OCR轉CSV
        if toCSV:
            # 標頭資訊(重要項目)
            Header = (' ', 'PN', 'Date', 'QTY', 'LOT', 'COO')

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

                List = ['-', '-', '-', '-', '-']
                EID = 0  # 換行用
                #     s = str(path)
                #     List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                for line in result:
                    line2 = line
                    line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割

                    # PN
                    if ('PN' in line2[0] or 'P/N' in line2[0]) and overwrite[PN] == 0:
                        if len(line2[0]) > 1 and len(line2) == 2:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][2:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0


                    # Date
                    elif ('Date' in line2[0] or 'DATE' in line2[0]) and overwrite[
                        Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                        if len(line2) > 1 and len(line2) == 2:
                            List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[Date] = line2[0][4:].lstrip(' ')
                        overwrite[Date] = 1  # 填完了
                        EID = 0  # 不用換行

                    # QTY
                    elif ('Qty' in line2[0] or 'QTY' in line2[0] or 'quantity' in line2[0] or 'Quantity' in line2[
                        0]) and overwrite[QTY] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[QTY] = 1
                        EID = 0

                    # LOT
                    elif ('LOT' in line2[0] or 'Lot' in line2[0]) and overwrite[LOT] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[LOT] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[LOT] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[LOT] = 1
                        EID = 0

                    # COO
                    elif ('COO' in line2[0] or 'Coo' in line2[0] or 'CoO' in line2[0] or 'Country' in line2[0]) and \
                            overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line2[0] or 'C.o.o.' in line2[0]) and overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('MADE IN' in line2[0] or 'Made In' in line2[0]) and overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][7:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0

                #######################################
                overwrite[0] = 1
                print(List)
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite) - 2):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)  # 印出來




        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start
        print("************************")
        print(f"執行時間: {exe_time:.4}")
        print("************************")


    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")

def photo_obj_detection(model_path,GPU_ratio=0.6,toCSV=True):

    # ----YOLO v4 init
    global os
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    # pathlist = sorted(Path("./input_dir/HD_img/").glob('*'))  # 用哪個資料夾裡的檔案
    pathlist = sorted(Path("./input_dir/Test_img/").glob('*'))  # 用哪個資料夾裡的檔案

    for path in pathlist:  # path每張檔案的路徑


        img_path = os.path.join('.', path)
        print(img_path)
        # ----YOLO v4 variable init
        img = cv2.imread(img_path)

        # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
        start = time.time()

        # 做sha_crap前處理
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mod_img = modify_contrast_and_brightness2(img, 0, 50)  # 調整圖片對比
        # ret, th1 = cv2.threshold(mod_img, 120, 255, cv2.THRESH_BINARY)  # 二值化圖片
        # img = sharpen(mod_img, th1, 0.6)  # 疊加二值化圖片與調完對比之圖片 0.6為兩圖佔比

        # googleOCR辨識

        # 設置API位置
        import os
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Google_json_path
        image_path = r'./result_dir/result_pic_yolo_crop.jpg'

        # 設定辨識function
        def google_detect_text(path):
            """Detects text in the file."""
            from google.cloud import vision
            import io
            client = vision.ImageAnnotatorClient()

            with io.open(path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            response = client.text_detection(image=image)
            texts = response.text_annotations
            print('Texts:')

            # 建立result_list 儲存辨識結果
            result_list = texts[0].description.split('\n')

            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))
            # 回傳辨識結果
            return result_list

        result = google_detect_text(img_path)

        print("Text Part:\n")
        for res in result:
            print(res)

        # ----YOLO v4 detection-----------------
        # yolo_img, pyz_decoded_str = yolo_v4.detection(img)
        # decode_result = pyz_decoded_str
        # # 印出Barcode/QRCode內容
        # print("Barcode/QRCode Part:\n\n")
        # if decode_result != []:
        #     for res in decode_result:
        #         print(res)
        # else:
        #     print("Decode Fail")

        ####################################################

        # 儲存照片路徑
        # result_img_path = "./result_dir/"+str(path)+".jpg"
        # # 儲存yolo辨識照片
        # cv2.imwrite(result_img_path, yolo_img)
        #####################################################
        # OCR轉CSV
        if toCSV:
            # 標頭資訊(重要項目)
            Header = (' ', 'PN', 'Date', 'QTY', 'LOT', 'COO')

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

                List = ['-', '-', '-', '-', '-']
                EID = 0  # 換行用
                #     s = str(path)
                #     List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                for line in result:
                    line2 = line
                    line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割

                    # PN
                    if ('PN' in line2[0] or 'P/N' in line2[0]) and overwrite[PN] == 0:
                        if len(line2[0]) > 1 and len(line2) == 2:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][2:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0


                    # Date
                    elif ('Date' in line2[0] or 'DATE' in line2[0]) and overwrite[
                        Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                        if len(line2) > 1 and len(line2) == 2:
                            List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[Date] = line2[0][4:].lstrip(' ')
                        overwrite[Date] = 1  # 填完了
                        EID = 0  # 不用換行

                    # QTY
                    elif ('Qty' in line2[0] or 'QTY' in line2[0] or 'quantity' in line2[0] or 'Quantity' in line2[
                        0]) and overwrite[QTY] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[QTY] = 1
                        EID = 0

                    # LOT
                    elif ('LOT' in line2[0] or 'Lot' in line2[0]) and overwrite[LOT] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[LOT] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[LOT] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[LOT] = 1
                        EID = 0

                    # COO
                    elif ('COO' in line2[0] or 'Coo' in line2[0] or 'CoO' in line2[0] or 'Country' in line2[0]) and \
                            overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line2[0] or 'C.o.o.' in line2[0]) and overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('MADE IN' in line2[0] or 'Made In' in line2[0]) and overwrite[COO] == 0:
                        if len(line2) > 1 and len(line2) == 2:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][7:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0

                #######################################
                overwrite[0] = 1
                print(List)
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite) - 2):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)  # 印出來



        end = time.time()

        # 用start - end算出程式運行時間，並且print出來
        exe_time = end - start
        print("************************")
        print(f"執行時間: {exe_time:.4}")
        print("************************")


    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")

def cross_photo_obj_detection(model_path, GPU_ratio=0.6,toCSV=True):
    # ----YOLO v4 init
    yolo_v4 = Yolo_v4(model_path, GPU_ratio=GPU_ratio)

    # 讀取top照片
    img_top = cv2.imread('./input_dir/cross_img_fold/cross_img_top.png')

    # 讀取side照片
    img_side = cv2.imread('./input_dir/cross_img_fold/cross_img_side.png')

    # YOLO v4 detection(TOP)
    yolo_top_img, pyz_decoded_top_str = yolo_v4.detection(img_top)

    # YOLO v4 detection(side)
    yolo_side_img, pyz_decoded_side_str = yolo_v4.detection(img_side)



    # googleOCR辨識

    # 設置API位置
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =Google_json_path
    top_image_path = r'./input_dir/cross_img_fold/cross_img_top.png'
    side_image_path = r'./input_dir/cross_img_fold/cross_img_side.png'

    # 設定辨識function
    def google_detect_text(path):
        """Detects text in the file."""
        from google.cloud import vision
        import io
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        print('Texts:')

        # 建立result_list 儲存辨識結果
        result_list = texts[0].description.split('\n')

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        # 回傳辨識結果
        return result_list

    top_result = google_detect_text(top_image_path)
    side_result = google_detect_text(side_image_path)

    result = top_result+side_result

    print("Text Part:\n")
    for res in result:
        print(res)

    # zbar decode
    decode_result_top = pyz_decoded_top_str
    decode_result_side = pyz_decoded_side_str
    decode_result = decode_result_top+decode_result_side

    # 匯出辨識結果(txt)
    ocr_result_path = './result_dir/result_txt.txt'
    decode_result_path = './result_dir/decode_result_txt.txt'
    f = open(ocr_result_path, 'w')
    fc = open(decode_result_path, 'w')


    # 印出result字元
    print("Text Part:\n")
    for res in result:
        f.write(res + '\n')
        print(res)

    # 印出Barcode/QRCode內容
    print("Barcode/QRCode Part:\n\n")
    if decode_result != []:
        for res in decode_result:
            fc.write(res + '\n')
            print(res)
    else:
        print("Decode Fail")
    # OCR轉CSV
    if toCSV:
        # 標頭資訊(重要項目)
        Header = (' ', 'PN', 'Date', 'QTY', 'LOT', 'COO')

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

            List = ['-', '-', '-', '-', '-']
            EID = 0  # 換行用
            #     s = str(path)
            #     List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
            overwrite = [0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
            for line in result:
                line2 = line
                line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割

                # PN
                if ('PN' in line2[0] or 'P/N' in line2[0]) and overwrite[PN] == 0:
                    if len(line2[0]) > 1 and len(line2) == 2:
                        List[PN] = line2[1].lstrip(' ')
                    else:
                        List[PN] = line2[0][2:].lstrip(' ')
                    overwrite[PN] = 1
                    EID = 0


                # Date
                elif ('Date' in line2[0] or 'DATE' in line2[0]) and overwrite[
                    Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                    if len(line2) > 1 and len(line2) == 2:
                        List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                    else:
                        List[Date] = line2[0][4:].lstrip(' ')
                    overwrite[Date] = 1  # 填完了
                    EID = 0  # 不用換行

                # QTY
                elif ('Qty' in line2[0] or 'QTY' in line2[0] or 'quantity' in line2[0] or 'Quantity' in line2[
                    0]) and overwrite[QTY] == 0:
                    if len(line2) > 1 and len(line2) == 2:
                        List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                    else:
                        List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                    overwrite[QTY] = 1
                    EID = 0

                # LOT
                elif ('LOT' in line2[0] or 'Lot' in line2[0]) and overwrite[LOT] == 0:
                    if len(line2) > 1 and len(line2) == 2:
                        List[LOT] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                    else:
                        List[LOT] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                    overwrite[LOT] = 1
                    EID = 0

                # COO
                elif ('COO' in line2[0] or 'Coo' in line2[0] or 'CoO' in line2[0] or 'Country' in line2[0]) and \
                        overwrite[COO] == 0:
                    if len(line2) > 1 and len(line2) == 2:
                        List[COO] = line2[1].lstrip(' ')
                    else:
                        List[COO] = line2[0][3:].lstrip(' ')
                    overwrite[COO] = 1
                    EID = 0
                elif ('C.O.O.' in line2[0] or 'C.o.o.' in line2[0]) and overwrite[COO] == 0:
                    if len(line2) > 1 and len(line2) == 2:
                        List[COO] = line2[1].lstrip(' ')
                    else:
                        List[COO] = line2[0][6:].lstrip(' ')
                    overwrite[COO] = 1
                    EID = 0
                elif ('MADE IN' in line2[0] or 'Made In' in line2[0]) and overwrite[COO] == 0:
                    if len(line2) > 1 and len(line2) == 2:
                        List[COO] = line2[1].lstrip(' ')
                    else:
                        List[COO] = line2[0][7:].lstrip(' ')
                    overwrite[COO] = 1
                    EID = 0

            #######################################
            overwrite[0] = 1
            print(List)
            if decode_result != []:
                wrote = decode_result
                for a in range(len(overwrite) - 2):
                    if overwrite[a] == 0:
                        for res in range(len(decode_result)):
                            if wrote[res] != 'wrote' and decode_result[res] != '':
                                print(decode_result[res])
                                List[a] = decode_result[res]
                                overwrite[a] = 1
                                wrote[res] = 'wrote'
                                break
            writer.writerow(List)  # 印出來

    #####################################################
    # ----release
    f.close()
    fc.close()
    yolo_v4.sess.close()
    cv2.destroyAllWindows()

def real_time_obj_detection_chioce(model_path,GPU_ratio=0.8):
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
        fc = open(decode_result_path, 'w')


        if ret is True:
            #----YOLO v4 detection
            yolo_img,pyz_decoded_str = yolo_v4.detection(img)

            # 在錄影的過程中儲存解碼內容
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
            cv2.imshow("YOLO v4 by JohnnyAI", yolo_img)

            # #----image writing
            # if writer is not None:
            #     writer.write(yolo_img)


            # ----按下Q鍵拍下、儲存一張照片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #####################################################
                # 儲存原始照片
                cv2.imwrite('./result_dir/result_pic_orig.jpg', pic)
                # input("Please press the Enter key to proceed")
                # 儲存yolo辨識照片
                cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)
                # input("Please press the Enter key to proceed")
                #####################################################

                # csv分類


                # 公司/項目
                THALES = (
                ' ', 'Company', 'Date', 'Po no', 'PN', 'Batch#', 'First ID', 'Last ID', 'Quantity', 'COO', 'Sleeve#',
                'BOX#')
                EDOM = (
                ' ', 'Company', 'Gemalto PN', 'EDOM PN', 'LOT#', 'Date code', 'Quantity', 'COO', 'MSL', 'BOX#', 'REEL#')
                SkyTra = (' ', 'Company', 'PART ID', 'D/C', 'QTY', 'Bin', 'Date')
                AKOUSTIS = (' ', 'Company', 'Part#', 'LOT#', 'MFG#', 'DTE', 'QTY')
                Silicon = (' ', 'Company', 'Country', 'SUPPLIER', 'DATECODE', 'QTY', 'CODE', 'SEALDATE')
                # CSV

                # ***********************************************************************
                # 從這邊開始讀取拍攝到的照片並作OCR辨識
                # ***********************************************************************

                # 用time的套件紀錄開始辨識的時間(用於計算程式運行時間)
                start = time.time()

                # 讀取拍攝好的照片(result_pic_orig.jpg)
                Comp = 0  # 公司
                img_path = './result_dir/result_pic_orig.jpg'  # 用這個路徑讀取最後拍下的照片
                # ----YOLO v4 variable init
                img = cv2.imread(img_path)



                # 做retinex前處理)
                # with open('config.json', 'r') as f:
                #     config = json.load(f)

                # 共有三種模式

                # msrcr處理
                # img = retinex.MSRCR(
                #     img,
                #     config['sigma_list'],
                #     config['G'],
                #     config['b'],
                #     config['alpha'],
                #     config['beta'],
                #     config['low_clip'],
                #     config['high_clip']
                # )

                # 做sha_crap前處理
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mod_img = modify_contrast_and_brightness2(img, 0, 50)  # 調整圖片對比
                ret, th1 = cv2.threshold(mod_img, 120, 255, cv2.THRESH_BINARY)  # 二值化圖片
                img = sharpen(mod_img, th1, 0.6)  # 疊加二值化圖片與調完對比之圖片 0.6為兩圖佔比
                cv2.imwrite('./result_dir/result_pic_yolo_crap_sha.jpg', img)

                # # amsrcr處理
                # img = retinex.automatedMSRCR(
                #     img,
                #     config['sigma_list']
                # )
                #
                # # msrcp處理
                # img = retinex.MSRCP(
                #     img,
                #     config['sigma_list'],
                #     config['low_clip'],
                #     config['high_clip']
                # )


                # 將yolo找到的code部分刪掉
                # 讀取yolo找到的座標
                with open(r'.\result_dir\yolo_box.txt', 'r') as f:
                    coordinates = f.read()
                spilt_coordinates = coordinates.split("\n")

                # 切掉各個code的區域
                for coordinate in spilt_coordinates:
                    if len(coordinate.split(",")) > 1:
                        x_min = int(coordinate.split(",")[0])
                        x_max = int(coordinate.split(",")[1])
                        y_min = int(coordinate.split(",")[2])
                        y_max = int(coordinate.split(",")[3])

                        padding_x = 5
                        padding_y = 8

                        # x padding
                        if (x_max - x_min > 2 * padding_x):
                            x_max -= padding_x
                            x_min += padding_x

                        # y padding
                        if (y_max - y_min > 2 * padding_y):
                            y_max -= padding_y
                            y_min += padding_y

                        # 轉換x_min,x_max,y_min,y_max為x_left,y_top,w,h
                        start_point = (x_min, y_min)
                        end_point = (x_max, y_max)
                        color = (0, 0, 0)
                        # Thickness of -1 will fill the entire shape
                        thickness = -1

                        img = cv2.rectangle(img, start_point, end_point, color, thickness)
                # 儲存yolo_crop照片
                cv2.imwrite('./result_dir/result_pic_yolo_crop.jpg', img)
                print("***************************")

                # googleOCR辨識

                # 設置API位置
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Google_json_path
                image_path = r'./result_dir/result_pic_yolo_crop.jpg'

                # 設定辨識function
                def google_detect_text(path):
                    """Detects text in the file."""
                    from google.cloud import vision
                    import io
                    client = vision.ImageAnnotatorClient()

                    with io.open(path, 'rb') as image_file:
                        content = image_file.read()

                    image = vision.Image(content=content)

                    response = client.text_detection(image=image)
                    texts = response.text_annotations
                    print('Texts:')

                    # 建立result_list 儲存辨識結果
                    result_list = texts[0].description.split('\n')

                    if response.error.message:
                        raise Exception(
                            '{}\nFor more info on error messages, check: '
                            'https://cloud.google.com/apis/design/errors'.format(
                                response.error.message))
                    # 回傳辨識結果
                    return result_list

                result = google_detect_text(img_path)



                # 匯出辨識結果(txt)
                result_path = './result_dir/result_txt.txt'
                # decode_result_path = './result_dir/decode_result_txt.txt'
                f = open(result_path, 'w')
                # fc = open(decode_result_path, 'w')



                # 印出PaddleOCR結果
                print("Text Part:\n")
                for res in result:
                    f.write(res + '\n')
                    print(res)


                # 印出Barcode/QRCode內容
                print("Barcode/QRCode Part:\n")
                for decode in decode_list:
                    fc.write(decode+'\n')
                    print(decode)
                decode_list = []


                #################tag查詢功能開始#################
                data = []
                col = []
                spilt_s = []
                num = input("總欄位數量: ")  # excel中要有多少個欄位

                List = []
                for i in range(int(num)):
                    List.append('-')

                index = 0

                for i in range(int(num)):
                    col_tmp = input("輸入欄位名稱: ")  # 每個欄位的名稱
                    spilt_tmp = input("輸入分隔符號，若無分隔符號則輸入欄位最後一個字母: ")
                    col.append(col_tmp)
                    spilt_s.append(spilt_tmp)
                    feat = []
                    write, pre = 0, 0
                    for res in result:
                        res1 = res
                        res1 = res1.split(spilt_tmp)
                        if write == 1:
                            write = 0
                            List[pre] = res1[0]
                            index += 1
                        if compare(col_tmp, res):
                            if len(res1[1]) > 1:
                                List[index] = res1[1]
                                index += 1
                            else:
                                write = 1
                                pre = index

                print(List)
                #################tag查詢功能結束#################

                end = time.time()

                # 用start - end算出程式運行時間，並且print出來
                exe_time = end - start
                print("************************")
                print(f"執行時間: {exe_time:.4}")
                print("************************")
                #####################################################
                # ----release
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


    # if writer is not None:
    #     writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = r".\yolov4-obj_best_416.ckpt.meta"
    # model_path = r"C:\Users\shiii\YOLO_v4-master\yolov4_416.ckpt.meta"
    GPU_ratio = 0.8
    # real_time_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)
    # real_time_obj_detection_chioce(model_path, GPU_ratio=GPU_ratio)
    photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=False)
    # photo_obj_detection_HD(model_path,GPU_ratio=GPU_ratio,toCSV=False)
    # cross_photo_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True)