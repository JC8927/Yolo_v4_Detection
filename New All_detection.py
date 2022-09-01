import cv2,time
import numpy
import numpy as np
from numpy import number
import tensorflow
from src.YOLO import YOLO
from src.Feature_parse_tf import get_predict_result
from utils import tools
import csv
from paddleocr import PaddleOCR,draw_ocr
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import copy
import re
import xlwt
from xlwt import Workbook


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
    cap = cv2.VideoCapture(0)
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
        print("boxes: ",boxes)
        print("score:",score)
        img_bgr ,decoded_str= tools.draw_img(img_bgr, boxes, score, label, self.label_dict, self.color_table)

        return img_bgr,decoded_str




def real_time_obj_detection(model_path,GPU_ratio=0.2):
    #----var
    frame_count = 0
    FPS = "0"
    d_t = 0

    #----video streaming init
    cap, height, width, writer = video_init()

    #----YOLO v4 init
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)

    while (cap.isOpened()):

        ret, img = cap.read()
        pic = numpy.array(img)
        if ret is True:
            #----YOLO v4 detection
            yolo_img,pyz_decoded_str = yolo_v4.detection(img)

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

            #----image writing
            if writer is not None:
                writer.write(yolo_img)

            #----進行偵測
            if cv2.waitKey(1) & 0xFF == ord('a'):
                # 儲存原始照片
                cv2.imwrite('./result_dir/result_pic_orig.jpg', pic)

                # 儲存yolo辨識照片
                cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)

                # 將yolo找到的code部分刪掉
                # 讀取yolo找到的座標
                with open(r'.\result_dir\yolo_box.txt', 'r') as f:
                    coordinates = f.read()
                spilt_coordinates = coordinates.split("\n")
                img = pic
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

                        print(x_min)
                        print(x_max)
                        print(y_min)
                        print(y_max)
                        print()
                        # 轉換x_min,x_max,y_min,y_max為x_left,y_top,w,h
                        start_point = (x_min, y_min)
                        end_point = (x_max, y_max)
                        color = (0, 0, 0)
                        # Thickness of -1 will fill the entire shape
                        thickness = -1

                        print(start_point)
                        print(end_point)

                        img = cv2.rectangle(img, start_point, end_point, color, thickness)
                # 儲存yolo_crop照片
                cv2.imwrite('./result_dir/result_pic_yolo_crop.jpg', img)
                print("***************************")

                #####################################################
                # paddleOCR辨識
                ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
                img_path = './result_dir/result_pic_orig.jpg'
                result = ocr.ocr(img_path, cls=False)
                decode_result = pyz_decoded_str
                # 匯出辨識結果(txt)
                result_path = './result_dir/result_txt.txt'
                decode_result_path = './result_dir/decode_result_txt.txt'
                f = open(result_path, 'w')
                fc = open(decode_result_path, 'w')
                # 印出字元與位置
                '''for line in result:
                    f.write(res[1]+'\n')
                    print(line)'''
                # 印出字元
                print("Text Part:\n")
                for res in result:
                    f.write(res[1][0] + '\n')
                    print(res[1][0])
                # 印出Barcode/QRCode內容
                print("Barcode/QRCode Part:\n")
                if decode_result != []:
                    for res in decode_result:
                        fc.write(res + '\n')
                        print(res)
                else:
                    print("Decode Fail")

            # ----按下Q鍵停止錄影，並拍下、儲存一張照片
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("get image failed")
            break

    #####################################################
    # 讓相片停留
    cv2.imshow("YOLO v4 by JohnnyAI", yolo_img)
    # 儲存原始照片
    cv2.imwrite('./result_dir/result_pic_orig.jpg', pic)
    # input("Please press the Enter key to proceed")
    # 儲存yolo辨識照片
    cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)
    # input("Please press the Enter key to proceed")
    #####################################################
    # paddleOCR辨識
    ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
    img_path = './result_dir/result_pic_orig.jpg'
    result = ocr.ocr(img_path, cls=False)
    decode_result = pyz_decoded_str
    # 匯出辨識結果(txt)
    result_path = './result_dir/result_txt.txt'
    decode_result_path = './result_dir/decode_result_txt.txt'
    f = open(result_path, 'w')
    fc = open(decode_result_path, 'w')
    # 印出字元與位置
    '''for line in result:
        f.write(res[1]+'\n')
        print(line)'''
    # 印出字元
    print("Text Part:\n")
    for res in result:
        f.write(res[1][0] + '\n')
        print(res[1][0])
    # 印出Barcode/QRCode內容
    print("Barcode/QRCode Part:\n")
    if decode_result != []:
        for res in decode_result:
            fc.write(res + '\n')
            print(res)
    else:
        print("Decode Fail")
    #####################################################
    # csv分類
    # 資料夾裡面每個檔案
    pathlist = ['./result_dir/result_pic_orig.jpg']  # 用哪個資料夾裡的檔案

    # 公司/項目
    THALES = (' ', 'Company', 'Date', 'Po no', 'PN', 'Batch#', 'First ID', 'Last ID', 'Quantity', 'COO', 'Sleeve#', 'BOX#')
    EDOM = (' ', 'Company', 'Gemalto PN', 'EDOM PN', 'LOT#', 'Date code', 'Quantity', 'COO', 'MSL', 'BOX#', 'REEL#')
    SkyTra = (' ', 'Company', 'PART ID', 'D/C', 'QTY', 'Bin', 'Date')
    AKOUSTIS = (' ', 'Company', 'Part#', 'LOT#', 'MFG#', 'DTE', 'QTY')
    Silicon = (' ', 'Company', 'Country', 'SUPPLIER', 'DATECODE', 'QTY', 'CODE', 'SEALDATE')
    # CSV



    for path in pathlist:  # path每張檔案的路徑

        Comp = 0  # 公司
        img_path = './result_dir/result_pic_orig.jpg' # 用這個路徑讀取最後拍下的照片
        # ----YOLO v4 variable init
        img = cv2.imread(img_path)

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

                print(x_min)
                print(x_max)
                print(y_min)
                print(y_max)
                print()
                # 轉換x_min,x_max,y_min,y_max為x_left,y_top,w,h
                start_point = (x_min, y_min)
                end_point = (x_max, y_max)
                color = (0, 0, 0)
                # Thickness of -1 will fill the entire shape
                thickness = -1

                print(start_point)
                print(end_point)

                img = cv2.rectangle(img, start_point, end_point, color, thickness)
        # 儲存yolo_crop照片
        cv2.imwrite('./result_dir/result_pic_yolo_crop.jpg', img)
        print("***************************")

        # ---paddleOCR detection-----------------
        ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
        result = ocr.ocr(img_path, cls=False)  # OCR

        print("Text Part:\n")
        for res in result:
            print(res[1][0])

        # ----YOLO v4 detection-----------------
        yolo_img, pyz_decoded_str = yolo_v4.detection(img)
        decode_result = pyz_decoded_str
        # 印出Barcode/QRCode內容
        print("Barcode/QRCode Part:\n\n")


        # 不要跳Decode Fail訊息

        # if decode_result != []:
        #     for res in decode_result:
        #         print(res)
        # else:
        #     print("Decode Fail")
        #####################################################

        # 儲存照片路徑
        # result_img_path = "./result_dir/"+str(path)+".jpg"
        # # 儲存yolo辨識照片
        # cv2.imwrite(result_img_path, yolo_img)
        #####################################################

        # 找是哪家公司
        for line in result:
            if 'THALES' in line[1][0]:
                Comp = 1
                break
            elif 'EDOM' in line[1][0]:
                Comp = 2
                break
            elif 'SkyT' in line[1][0]:
                Comp = 3
                break
            elif 'Silicon' in line[1][0]:
                Comp = 4
                break
            elif 'AKOUSTIS' in line[1][0]:
                Comp = 5
                break

        # 檢查是否存在各公司資料夾，不存在的話就創立一個新的(包含標頭)
        if not os.path.isfile('./result_dir/Company_OCR/THALES_csv.csv'):
            with open('./result_dir/Company_OCR/THALES_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(THALES)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/EDOM_csv.csv'):
            with open('./result_dir/Company_OCR/EDOM_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(EDOM)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/SkyTra_csv.csv'):
            with open('./result_dir/Company_OCR/SkyTra_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(SkyTra)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/Silicon_csv.csv'):
            with open('./result_dir/Company_OCR/Silicon_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(Silicon)  # 列出公司有的項目 (之後看寫在哪 只用跑一次)
        if not os.path.isfile('./result_dir/Company_OCR/AKOUSTIS_csv.csv'):
            with open('./result_dir/Company_OCR/AKOUSTIS_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(AKOUSTIS)  # 列出公司有的項目

        result_list=[]

        # for line in result:
        #     line2 = line[1][0]
        #     print(line2)
        #     result_list.append(line2)
        #
        # print("list content:\n")
        # print(result_list)

        # THALES
        if (Comp == 1):
            result_path = './result_dir/Company_OCR/THALES_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Date, Po, PN, Batch, FirstE, LastE, QTY, COO, Sle, BOX = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11  # 哪一項放在第幾格
                List = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0  # 換行用
                s = str(path)
                List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割

                    # Date
                    if ('Date' in line[1][0] or 'DATE' in line[1][0]) and overwrite[Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                        if len(line2) > 1:
                            List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[Date] = line2[0][4:].lstrip(' ')
                        overwrite[Date] = 1  # 填完了
                        EID = 0  # 不用換行

                    # Company
                    elif 'THALES' in line[1][0] and overwrite[Company] == 0:  # 那行有公司名
                        List[Company] = 'THALES'  # 填公司名
                        overwrite[Company] = 1  # 填了
                        EID = 0  # 不用換行

                    elif ('PO No.' in line[1][0]) or 'P.O. #' in line[1][0] and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][6:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行
                    elif ('PONo.' in line[1][0] or 'P.O.#' in line[1][0]) and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][5:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行
                    elif ('P.O#' in line[1][0]) and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][4:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行

                    # PN
                    elif ('PN' in line[1][0]) and overwrite[PN] == 0:
                        if len(line2) > 1:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][2:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0
                    elif ('P/N' in line[1][0]) and overwrite[PN] == 0:
                        if len(line2) > 1:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][3:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0

                    # Batch
                    elif 'Batch' in line[1][0] and overwrite[Batch] == 0:
                        if len(line2) > 1: List[Batch] = line2[1].lstrip(' ')
                        overwrite[Batch] = 1
                        EID = 0

                    # EID(換行)
                    elif 'First EID' in line[1][0]:
                        EID = 1  # 這行沒東西 換行
                    elif 'Last EID' in line[1][0]:
                        EID = 2  # 這行沒東西 換行
                    elif 'First ICCID' in line[1][0]:
                        if len(line2) > 1:
                            List[FirstE] = line2[1].lstrip(' ')
                        else:
                            List[FirstE] = line2[0][11:].lstrip(' ')
                        overwrite[FirstE] = 1  # 填了
                        EID = 0  # 不用換行
                    elif 'Last ICCID' in line[1][0]:
                        if len(line2) > 1:
                            List[LastE] = line2[1].lstrip(' ')
                        else:
                            List[LastE] = line2[0][10:].lstrip(' ')
                        overwrite[LastE] = 1
                        EID = 0

                    # QTY
                    elif ('Qty' in line[1][0] or 'QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[QTY] = 1
                        EID = 0

                    # COO
                    elif ('COO' in line[1][0] or 'Coo' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line[1][0] or 'C.o.o.' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('MADE IN' in line[1][0] or 'Made In' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][7:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0

                    # SLEEVE
                    elif 'Sleeve' in line[1][0] and overwrite[Sle] == 0:
                        if len(line2) > 1: List[Sle] = line2[1].lstrip(' ')
                        overwrite[Sle] = 1
                        EID = 0
                    elif ('BOX #' in line[1][0] or 'Box #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][5:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX#' in line[1][0] or 'Box#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][4:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX' in line[1][0] or 'Box' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][3:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0

                    # EID
                    elif EID == 1 and overwrite[FirstE] == 0:  # 上一行測到讓EID變1的
                        List[FirstE] = line2[0].lstrip(' ')  # 填
                        overwrite[FirstE] = 1  # 填了
                        EID = 0  # 不用換行
                    elif EID == 2 and overwrite[LastE] == 0:  # 上一行測到讓EID變2的
                        List[LastE] = line2[0].lstrip(' ')
                        overwrite[LastE] = 1
                        EID = 0

                #######################################
                overwrite[0] = 1
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

        # EDOM
        elif (Comp == 2):
            result_path = './result_dir/Company_OCR/EDOM_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, GPN, EPN, Lot, DateCo, QTY, COO, MSL, BOX, REEL = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                List = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0
                s = str(path)
                List[0] = s.strip("/content/LABEL/")
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')
                    if ('Date co' in line[1][0] or 'DATE co' in line[1][0]) and overwrite[DateCo] == 0:
                        if len(line2) > 1: List[DateCo] = line2[1]
                        overwrite[DateCo] = 1
                        EID = 0
                    elif 'EDOM' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'EDOM'
                        overwrite[Company] = 1
                        EID = 0
                    elif ('Lot#' in line[1][0] or 'LOT#' in line[1][0]) and overwrite[Lot] == 0:
                        if len(line2) > 1:
                            List[Lot] = line2[1].lstrip(' ')
                        else:
                            List[Lot] = line2[0][4:].lstrip(' ')
                        overwrite[Lot] = 1
                        EID = 0
                    elif ('Lot' in line[1][0] or 'LOT' in line[1][0]) and overwrite[Lot] == 0:
                        if len(line2) > 1:
                            List[Lot] = line2[1].lstrip(' ')
                        else:
                            List[Lot] = line2[0][3:].lstrip(' ')
                        overwrite[Lot] = 1
                        EID = 0
                    elif ('Gemalto' in line[1][0] or 'A1') and overwrite[GPN] == 0:
                        if len(line2) > 1: List[GPN] = line2[1].lstrip(' ')
                        overwrite[GPN] = 1
                        EID = 0
                    elif ('EDOM PN' in line[1][0]) and overwrite[EPN] == 0:
                        if len(line2) > 1:
                            List[EPN] = line2[1].lstrip(' ')
                        else:
                            List[EPN] = line2[0][7:].lstrip(' ')
                        overwrite[EPN] = 1
                        EID = 0
                    elif ('EDOMPN' in line[1][0]) and overwrite[EPN] == 0:
                        if len(line2) > 1:
                            List[EPN] = line2[1].lstrip(' ')
                        else:
                            List[EPN] = line2[0][6:].lstrip(' ')
                        overwrite[EPN] = 1
                        EID = 0
                    elif ('Qty' in line[1][0] or 'QUAN' in line[1][0] or 'QTY' in line[1][0]) and overwrite[
                        QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(' I')
                        else:
                            List[QTY] = line2[0][3:].lstrip(' I')
                        overwrite[QTY] = 1
                        EID = 0
                    elif ('COO' in line[1][0] or 'Coo' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line[1][0] or 'C.o.o.' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('BOX #' in line[1][0] or 'Box #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][5:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX#' in line[1][0] or 'Box#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][4:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX' in line[1][0] or 'Box' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][3:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('REEL#' in line[1][0] or 'Reel#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][5:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('REEL' in line[1][0] or 'Reel' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][4:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('REEL #' in line[1][0] or 'Reel #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][6:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('MSL' in line[1][0] or 'msl' in line[1][0]) and overwrite[MSL] == 0:
                        if len(line2) > 1:
                            List[MSL] = line2[1].lstrip(' ')
                        else:
                            List[MSL] = line2[0][3:].lstrip(' ')
                        overwrite[MSL] = 1
                        EID = 0
                #######################################
                overwrite[0] = 1
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

        # SkyTra
        elif (Comp == 3):
            print(r"////////////////////////////////////")
            print("Comp = "+str(Comp))
            print(r"////////////////////////////////////")
            result_path = './result_dir/Company_OCR/SkyTra_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, PartID, DC, QTY, BIN, DATE = 1, 2, 3, 4, 5, 6
                List = ['-', '-', '-', '-', '-', '-', '-']
                s = str(path)
                List[0] = s.strip("./")
                overwrite = [0, 0, 0, 0, 0, 0, 0]
                write, pre = 0, 0
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')
                    if write == 1:
                        write = 0
                        List[pre] = line2[0]
                        overwrite[pre] = 1

                    if ('PARTID' in line[1][0] or 'PART ID' in line[1][0]) and overwrite[PartID] == 0:
                        if len(line2[1]) > 1:
                            List[PartID] = line2[1]
                            overwrite[PartID] = 1
                        elif overwrite[PartID] == 0:
                            write = 1
                            pre = PartID
                    elif 'SkyT' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'SkyTra'
                        overwrite[Company] = 1
                    elif ('D/C' in line[1][0]) and overwrite[DC] == 0:
                        if len(line2[1]) > 1:
                            List[DC] = line2[1].lstrip(' ')
                            overwrite[DC] = 1
                        elif overwrite[DC] == 0:
                            write = 1
                            pre = DC
                    elif ('QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2[1]) > 1:
                            List[QTY] = line2[1].lstrip(' ')
                            overwrite[QTY] = 1
                        elif overwrite[QTY] == 0:
                            write = 1
                            pre = QTY
                    elif ('Bin' in line[1][0]) and overwrite[BIN] == 0:
                        if len(line2[1]) > 1:
                            List[BIN] = line2[1].lstrip(' ')
                            overwrite[BIN] = 1
                        elif overwrite[BIN] == 0:
                            write = 1
                            pre = BIN
                    elif (('Date' in line[1][0] or 'ROHS' in line[1][0]) and overwrite[DATE] == 0):
                        if ('Date' in line[1][0] and len(line2[1]) > 1):
                            List[DATE] = line2[1].lstrip(' ')
                            overwrite[DATE] = 1
                        elif overwrite[DATE] == 0:
                            write = 1
                            pre = DATE

                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)

        # Silicon
        elif (Comp == 4):
            result_path = './result_dir/Company_OCR/Silicon_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Country, SUPPLIER, DATECODE, QTY, CODE, SEALDATE = 1, 2, 3, 4, 5, 6, 7
                List = ['-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0
                s = str(path)
                List[0] = s.strip(r"/content/LABEL/")
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0]
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')

                    if 'Silicon' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'Silicon Laboratories Inc.'
                        overwrite[Company] = 1
                        EID = 0
                    elif ('TW' in line[1][0] or 'CN' in line[1][0] or 'cN' in line[1][0] or 'cn' in line[1][
                        0] or 'Tw' in line[1][0]) and overwrite[Country] == 0:
                        List[Country] = line[1][0].lstrip(' AsemblinInd:')
                        overwrite[Country] = 1
                        EID = 0
                    elif ('SUPPLIER' in line[1][0] or 'ID' in line[1][0] or 'Customer' in line[1][0] or 'Part' in
                          line[1][0]) and overwrite[SUPPLIER] == 0:
                        if len(line2) > 1:
                            List[SUPPLIER] = line2[1].lstrip(' ')
                        else:
                            List[SUPPLIER] = line2[0][3:].lstrip(' ')
                        overwrite[SUPPLIER] = 1
                        EID = 0
                    elif ('DATECODE' in line[1][0] or 'Date Code' in line[1][0]) and overwrite[DATECODE] == 0:
                        if len(line2) > 1: List[DATECODE] = line2[1].lstrip(' ')
                        overwrite[DATECODE] = 1
                        EID = 0
                    elif ('Qty' in line[1][0] or 'QUAN' in line[1][0] or 'QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(r'QqTtYy() ')
                        else:
                            List[QTY] = line2[0][3:].lstrip(r'QqTtYy() ')
                        overwrite[QTY] = 1
                        EID = 0

                    elif (line[1][0].isdigit() and len(line[1][0]) > 9 or 'Trace Code' in line[1][0] or 'BOX' in
                          line[1][0]) and overwrite[CODE] == 0:
                        if len(line2) > 1:
                            List[CODE] = line2[1].lstrip(r' ')
                        else:
                            List[CODE] = line2[0].lstrip(' ')
                        overwrite[CODE] = 1
                        EID = 0

                    elif ('SEALDATE' in line[1][0] or 'Seal Date' in line[1][0] or 'SEAL DATE' in line[1][0]) and \
                            overwrite[SEALDATE] == 0:
                        if len(line2) > 1:
                            List[SEALDATE] = line2[1].lstrip(r' SEALDTealate')
                        else:
                            List[SEALDATE] = line[1][0].lstrip(r' SEALDTealate')
                        overwrite[SEALDATE] = 1
                        EID = 0
                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)

        # AKOUSTIS
        elif (Comp == 5):
            result_path = './result_dir/Company_OCR/AKOUSTIS_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Part, LOT, MFG, DTE, QTY = 1, 2, 3, 4, 5,6 # 哪一項放在第幾格
                List = ['-', '-', '-', '-', '-', '-', '-']
                EID = 0  # 換行用
                s = str(path)
                List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                index = 0
                for line in result:
                    line2 = line[1][0]
                    # print(line2)

                    # Company
                    if 'AKOUSTIS' in line[1][0] and overwrite[Company] == 0:  # 那行有公司名
                        List[Company] = 'AKOUSTIS'  # 填公司名
                        overwrite[Company] = 1  # 填了

                        EID = 0  # 不用換行

                    # Part#
                    elif ('Part' in line[1][0]) and overwrite[Part] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[Part] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        overwrite[Part] = 1  # 填了

                        EID = 0  # 不用換行

                    # LOT#
                    elif ('LOT' in line[1][0] or 'P/N' in line[1][0]) and overwrite[LOT] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[LOT] = line2[1].lstrip(' ')
                        overwrite[LOT] = 1

                        EID = 0

                    # MFG#
                    elif 'MFG' in line[1][0] and overwrite[MFG] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[MFG] = line2[1].lstrip(' ')
                        overwrite[MFG] = 1

                        EID = 0

                    # DTE
                    elif 'DTE' in line[1][0] and overwrite[DTE] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[DTE] = line2[1].lstrip(' ')
                        overwrite[DTE] = 1

                        EID = 0

                    # QTY
                    elif 'QTY' in line[1][0] and overwrite[QTY] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[QTY] = line2[1].lstrip(' ')
                        overwrite[QTY] = 1

                        EID = 0

                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
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
    cap.release()


    # if writer is not None:
    #     writer.release()
    cv2.destroyAllWindows()

def photo_obj_detection(model_path,GPU_ratio=0.8):

    # ----YOLO v4 init
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    print("yolo initial done")

    # 資料夾裡面每個檔案
    pathlist = sorted(Path("./input_dir/ALL_company/").glob('*'))  # 用哪個資料夾裡的檔案

    # 公司/項目
    THALES = (' ', 'Company', 'Date', 'Po no', 'PN', 'Batch#', 'First ID', 'Last ID', 'Quantity', 'COO', 'Sleeve#', 'BOX#')
    EDOM = (' ', 'Company', 'Gemalto PN', 'EDOM PN', 'LOT#', 'Date code', 'Quantity', 'COO', 'MSL', 'BOX#', 'REEL#')
    SkyTra = (' ', 'Company', 'PART ID', 'D/C', 'QTY', 'Bin', 'Date')
    AKOUSTIS = (' ', 'Company', 'Part#', 'LOT#', 'MFG#', 'DTE', 'QTY')
    Silicon = (' ', 'Company', 'Country', 'SUPPLIER', 'DATECODE', 'QTY', 'CODE', 'SEALDATE')
    # CSV



    for path in pathlist:  # path每張檔案的路徑

        Comp = 0  # 公司
        img_path = os.path.join('.', path)
        # ----YOLO v4 variable init
        img = cv2.imread(img_path)

        # # 將yolo找到的code部分刪掉
        # # 讀取yolo找到的座標
        # with open(r'.\result_dir\yolo_box.txt', 'r') as f:
        #     coordinates = f.read()
        # spilt_coordinates = coordinates.split("\n")
        #
        # # 切掉各個code的區域
        # for coordinate in spilt_coordinates:
        #     if len(coordinate.split(",")) > 1:
        #         x_min = int(coordinate.split(",")[0])
        #         x_max = int(coordinate.split(",")[1])
        #         y_min = int(coordinate.split(",")[2])
        #         y_max = int(coordinate.split(",")[3])
        #
        #         padding_x = 5
        #         padding_y = 8
        #
        #         # x padding
        #         if (x_max - x_min > 2 * padding_x):
        #             x_max -= padding_x
        #             x_min += padding_x
        #
        #         # y padding
        #         if (y_max - y_min > 2 * padding_y):
        #             y_max -= padding_y
        #             y_min += padding_y
        #
        #         print(x_min)
        #         print(x_max)
        #         print(y_min)
        #         print(y_max)
        #         print()
        #         # 轉換x_min,x_max,y_min,y_max為x_left,y_top,w,h
        #         start_point = (x_min, y_min)
        #         end_point = (x_max, y_max)
        #         color = (0, 0, 0)
        #         # Thickness of -1 will fill the entire shape
        #         thickness = -1
        #
        #         print(start_point)
        #         print(end_point)
        #
        #         img = cv2.rectangle(img, start_point, end_point, color, thickness)
        # # 儲存yolo_crop照片
        # cv2.imwrite('./result_dir/result_pic_yolo_crop.jpg', img)
        # print("***************************")
        # #####################################################

        # ---paddleOCR detection-----------------
        ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
        result = ocr.ocr(img_path, cls=False)  # OCR

        print("Text Part:\n")
        for res in result:
            print(res[1][0])

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
        #####################################################

        # 儲存照片路徑
        # result_img_path = "./result_dir/"+str(path)+".jpg"
        # # 儲存yolo辨識照片
        # cv2.imwrite(result_img_path, yolo_img)
        #####################################################

        # 找是哪家公司
        for line in result:
            if 'THALES' in line[1][0]:
                Comp = 1
                break
            elif 'EDOM' in line[1][0]:
                Comp = 2
                break
            elif 'SkyT' in line[1][0]:
                Comp = 3
                break
            elif 'Silicon' in line[1][0]:
                Comp = 4
                break
            elif 'AKOUSTIS' in line[1][0]:
                Comp = 5
                break

        # 檢查是否存在各公司資料夾，不存在的話就創立一個新的(包含標頭)
        if not os.path.isfile('./result_dir/Company_OCR/THALES_csv.csv'):
            with open('./result_dir/Company_OCR/THALES_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(THALES)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/EDOM_csv.csv'):
            with open('./result_dir/Company_OCR/EDOM_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(EDOM)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/SkyTra_csv.csv'):
            with open('./result_dir/Company_OCR/SkyTra_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(SkyTra)  # 列出公司有的項目
        if not os.path.isfile('./result_dir/Company_OCR/Silicon_csv.csv'):
            with open('./result_dir/Company_OCR/Silicon_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(Silicon)  # 列出公司有的項目 (之後看寫在哪 只用跑一次)
        if not os.path.isfile('./result_dir/Company_OCR/AKOUSTIS_csv.csv'):
            with open('./result_dir/Company_OCR/AKOUSTIS_csv.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(AKOUSTIS)  # 列出公司有的項目

        result_list=[]

        # for line in result:
        #     line2 = line[1][0]
        #     print(line2)
        #     result_list.append(line2)
        #
        # print("list content:\n")
        # print(result_list)

        # THALES
        if (Comp == 1):
            result_path = './result_dir/Company_OCR/THALES_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Date, Po, PN, Batch, FirstE, LastE, QTY, COO, Sle, BOX = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11  # 哪一項放在第幾格
                List = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0  # 換行用
                s = str(path)
                List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割

                    # Date
                    if ('Date' in line[1][0] or 'DATE' in line[1][0]) and overwrite[Date] == 0:  # 那行有Date Date那格沒被填過(有些公司有Date code又有Date ，Date code要寫前面)
                        if len(line2) > 1:
                            List[Date] = line2[1]  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[Date] = line2[0][4:].lstrip(' ')
                        overwrite[Date] = 1  # 填完了
                        EID = 0  # 不用換行

                    # Company
                    elif 'THALES' in line[1][0] and overwrite[Company] == 0:  # 那行有公司名
                        List[Company] = 'THALES'  # 填公司名
                        overwrite[Company] = 1  # 填了
                        EID = 0  # 不用換行

                    elif ('PO No.' in line[1][0]) or 'P.O. #' in line[1][0] and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][6:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行
                    elif ('PONo.' in line[1][0] or 'P.O.#' in line[1][0]) and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][5:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行
                    elif ('P.O#' in line[1][0]) and overwrite[Po] == 0:
                        if len(line2) > 1:
                            List[Po] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        else:
                            List[Po] = line2[0][4:].lstrip(' ')
                        overwrite[Po] = 1  # 填了
                        EID = 0  # 不用換行

                    # PN
                    elif ('PN' in line[1][0]) and overwrite[PN] == 0:
                        if len(line2) > 1:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][2:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0
                    elif ('P/N' in line[1][0]) and overwrite[PN] == 0:
                        if len(line2) > 1:
                            List[PN] = line2[1].lstrip(' ')
                        else:
                            List[PN] = line2[0][3:].lstrip(' ')
                        overwrite[PN] = 1
                        EID = 0

                    # Batch
                    elif 'Batch' in line[1][0] and overwrite[Batch] == 0:
                        if len(line2) > 1: List[Batch] = line2[1].lstrip(' ')
                        overwrite[Batch] = 1
                        EID = 0

                    # EID(換行)
                    elif 'First EID' in line[1][0]:
                        EID = 1  # 這行沒東西 換行
                    elif 'Last EID' in line[1][0]:
                        EID = 2  # 這行沒東西 換行
                    elif 'First ICCID' in line[1][0]:
                        if len(line2) > 1:
                            List[FirstE] = line2[1].lstrip(' ')
                        else:
                            List[FirstE] = line2[0][11:].lstrip(' ')
                        overwrite[FirstE] = 1  # 填了
                        EID = 0  # 不用換行
                    elif 'Last ICCID' in line[1][0]:
                        if len(line2) > 1:
                            List[LastE] = line2[1].lstrip(' ')
                        else:
                            List[LastE] = line2[0][10:].lstrip(' ')
                        overwrite[LastE] = 1
                        EID = 0

                    # QTY
                    elif ('Qty' in line[1][0] or 'QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料
                        else:
                            List[QTY] = line2[0][3:].lstrip(' ')  # 那行沒被分過(沒冒號) 刪掉前面3個字(QTY)
                        overwrite[QTY] = 1
                        EID = 0

                    # COO
                    elif ('COO' in line[1][0] or 'Coo' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line[1][0] or 'C.o.o.' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('MADE IN' in line[1][0] or 'Made In' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][7:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0

                    # SLEEVE
                    elif 'Sleeve' in line[1][0] and overwrite[Sle] == 0:
                        if len(line2) > 1: List[Sle] = line2[1].lstrip(' ')
                        overwrite[Sle] = 1
                        EID = 0
                    elif ('BOX #' in line[1][0] or 'Box #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][5:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX#' in line[1][0] or 'Box#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][4:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX' in line[1][0] or 'Box' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][3:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0

                    # EID
                    elif EID == 1 and overwrite[FirstE] == 0:  # 上一行測到讓EID變1的
                        List[FirstE] = line2[0].lstrip(' ')  # 填
                        overwrite[FirstE] = 1  # 填了
                        EID = 0  # 不用換行
                    elif EID == 2 and overwrite[LastE] == 0:  # 上一行測到讓EID變2的
                        List[LastE] = line2[0].lstrip(' ')
                        overwrite[LastE] = 1
                        EID = 0

                #######################################
                overwrite[0] = 1
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

        # EDOM
        elif (Comp == 2):
            result_path = './result_dir/Company_OCR/EDOM_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, GPN, EPN, Lot, DateCo, QTY, COO, MSL, BOX, REEL = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                List = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0
                s = str(path)
                List[0] = s.strip("/content/LABEL/")
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')
                    if ('Date co' in line[1][0] or 'DATE co' in line[1][0] or 'Datecode' in line[1][0]) and overwrite[DateCo] == 0:
                        if len(line2) > 1: List[DateCo] = line2[1]
                        overwrite[DateCo] = 1
                        EID = 0
                    elif 'EDOM' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'EDOM'
                        overwrite[Company] = 1
                        EID = 0
                    elif ('Lot#' in line[1][0] or 'LOT#' in line[1][0]) and overwrite[Lot] == 0:
                        if len(line2) > 1:
                            List[Lot] = line2[1].lstrip(' ')
                        else:
                            List[Lot] = line2[0][4:].lstrip(' ')
                        overwrite[Lot] = 1
                        EID = 0
                    elif ('Lot' in line[1][0] or 'LOT' in line[1][0]) and overwrite[Lot] == 0:
                        if len(line2) > 1:
                            List[Lot] = line2[1].lstrip(' ')
                        else:
                            List[Lot] = line2[0][3:].lstrip(' ')
                        overwrite[Lot] = 1
                        EID = 0
                    elif ('Gemalto' in line[1][0]  or 'A1') and overwrite[GPN] == 0:
                        if len(line2) > 1: List[GPN] = line2[1].lstrip(' ')
                        overwrite[GPN] = 1
                        EID = 0
                    elif ('EDOM PN' in line[1][0]) and overwrite[EPN] == 0:
                        if len(line2) > 1:
                            List[EPN] = line2[1].lstrip(' ')
                        else:
                            List[EPN] = line2[0][7:].lstrip(' ')
                        overwrite[EPN] = 1
                        EID = 0
                    elif ('EDOMPN' in line[1][0]) and overwrite[EPN] == 0:
                        if len(line2) > 1:
                            List[EPN] = line2[1].lstrip(' ')
                        else:
                            List[EPN] = line2[0][6:].lstrip(' ')
                        overwrite[EPN] = 1
                        EID = 0
                    elif ('Qty' in line[1][0] or 'QUAN' in line[1][0] or 'QTY' in line[1][0] or 'QY' in line[1][0]) and overwrite[
                        QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(' ')
                        else:
                            List[QTY] = line2[0][3:].lstrip(' |')
                        overwrite[QTY] = 1
                        EID = 0
                    elif ('COO' in line[1][0] or 'Coo' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][3:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('C.O.O.' in line[1][0] or 'C.o.o.' in line[1][0]) and overwrite[COO] == 0:
                        if len(line2) > 1:
                            List[COO] = line2[1].lstrip(' ')
                        else:
                            List[COO] = line2[0][6:].lstrip(' ')
                        overwrite[COO] = 1
                        EID = 0
                    elif ('BOX #' in line[1][0] or 'Box #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][5:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX#' in line[1][0] or 'Box#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][4:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('BOX' in line[1][0] or 'Box' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[BOX] = line2[1].lstrip(' ')
                        else:
                            List[BOX] = line2[BOX][3:].lstrip(' ')
                        overwrite[BOX] = 1
                        EID = 0
                    elif ('REEL#' in line[1][0] or 'Reel#' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][5:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('REEL' in line[1][0] or 'Reel' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][4:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('REEL #' in line[1][0] or 'Reel #' in line[1][0]) and overwrite[BOX] == 0:
                        if len(line2) > 1:
                            List[REEL] = line2[1].lstrip(' ')
                        else:
                            List[REEL] = line2[REEL][6:].lstrip(' ')
                        overwrite[REEL] = 1
                        EID = 0
                    elif ('MSL' in line[1][0] or 'msl' in line[1][0]) and overwrite[MSL] == 0:
                        if len(line2) > 1:
                            List[MSL] = line2[1].lstrip(' ')
                        else:
                            List[MSL] = line2[0][3:].lstrip(' ')
                        overwrite[MSL] = 1
                        EID = 0
                #######################################
                overwrite[0] = 1
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

        # SkyTra
        elif (Comp == 3):
            print(r"////////////////////////////////////")
            print("Comp = "+str(Comp))
            print(r"////////////////////////////////////")
            result_path = './result_dir/Company_OCR/SkyTra_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, PartID, DC, QTY, BIN, DATE = 1, 2, 3, 4, 5, 6
                List = ['-', '-', '-', '-', '-', '-', '-']
                s = str(path)
                List[0] = s.strip("./")
                overwrite = [0, 0, 0, 0, 0, 0, 0]
                write, pre = 0, 0
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')
                    if write == 1:
                        write = 0
                        List[pre] = line2[0]
                        overwrite[pre] = 1

                    if ('PARTID' in line[1][0] or 'PART ID' in line[1][0]) and overwrite[PartID] == 0:
                        if len(line2[1]) > 1:
                            List[PartID] = line2[1]
                            overwrite[PartID] = 1
                        elif overwrite[PartID] == 0:
                            write = 1
                            pre = PartID
                    elif 'SkyT' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'SkyTra'
                        overwrite[Company] = 1
                    elif ('D/C' in line[1][0]) and overwrite[DC] == 0:
                        if len(line2[1]) > 1:
                            List[DC] = line2[1].lstrip(' ')
                            overwrite[DC] = 1
                        elif overwrite[DC] == 0:
                            write = 1
                            pre = DC
                    elif ('QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2[1]) > 1:
                            List[QTY] = line2[1].lstrip(' ')
                            overwrite[QTY] = 1
                        elif overwrite[QTY] == 0:
                            write = 1
                            pre = QTY
                    elif ('Bin' in line[1][0]) and overwrite[BIN] == 0:
                        if len(line2[1]) > 1:
                            List[BIN] = line2[1].lstrip(' ')
                            overwrite[BIN] = 1
                        elif overwrite[BIN] == 0:
                            write = 1
                            pre = BIN
                    elif (('Date' in line[1][0] or 'ROHS' in line[1][0]) and overwrite[DATE] == 0):
                        if ('Date' in line[1][0] and len(line2[1]) > 1):
                            List[DATE] = line2[1].lstrip(' ')
                            overwrite[DATE] = 1
                        elif overwrite[DATE] == 0:
                            write = 1
                            pre = DATE

                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)

        # Silicon
        elif (Comp == 4):
            result_path = './result_dir/Company_OCR/Silicon_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Country, SUPPLIER, DATECODE, QTY, CODE, SEALDATE = 1, 2, 3, 4, 5, 6, 7
                List = ['-', '-', '-', '-', '-', '-', '-', '-']
                EID = 0
                s = str(path)
                List[0] = s.strip(r"/content/LABEL/")
                overwrite = [0, 0, 0, 0, 0, 0, 0, 0]
                for line in result:
                    line2 = line[1][0]
                    line2 = line2.split(':')

                    if 'Silicon' in line[1][0] and overwrite[Company] == 0:
                        List[Company] = 'Silicon Laboratories Inc.'
                        overwrite[Company] = 1
                        EID = 0
                    elif ('TW' in line[1][0] or 'CN' in line[1][0] or 'cN' in line[1][0] or 'cn' in line[1][
                        0] or 'Tw' in line[1][0]) and overwrite[Country] == 0:
                        List[Country] = line[1][0].lstrip(' AsemblinInd:')
                        overwrite[Country] = 1
                        EID = 0
                    elif ('SUPPLIER' in line[1][0] or 'ID' in line[1][0] or 'Customer' in line[1][0] or 'Part' in
                          line[1][0]) and overwrite[SUPPLIER] == 0:
                        if len(line2) > 1:
                            List[SUPPLIER] = line2[1].lstrip(' ')
                        else:
                            List[SUPPLIER] = line2[0][3:].lstrip(' ')
                        overwrite[SUPPLIER] = 1
                        EID = 0
                    elif ('DATECODE' in line[1][0] or 'Date Code' in line[1][0]) and overwrite[DATECODE] == 0:
                        if len(line2) > 1: List[DATECODE] = line2[1].lstrip(' ')
                        overwrite[DATECODE] = 1
                        EID = 0
                    elif ('Qty' in line[1][0] or 'QUAN' in line[1][0] or 'QTY' in line[1][0]) and overwrite[QTY] == 0:
                        if len(line2) > 1:
                            List[QTY] = line2[1].lstrip(r'QqTtYy() ')
                        else:
                            List[QTY] = line2[0][3:].lstrip(r'QqTtYy() ')
                        overwrite[QTY] = 1
                        EID = 0

                    elif (line[1][0].isdigit() and len(line[1][0]) > 9 or 'Trace Code' in line[1][0] or 'BOX' in
                          line[1][0]) and overwrite[CODE] == 0:
                        if len(line2) > 1:
                            List[CODE] = line2[1].lstrip(r' ')
                        else:
                            List[CODE] = line2[0].lstrip(' ')
                        overwrite[CODE] = 1
                        EID = 0

                    elif ('SEALDATE' in line[1][0] or 'Seal Date' in line[1][0] or 'SEAL DATE' in line[1][0]) and \
                            overwrite[SEALDATE] == 0:
                        if len(line2) > 1:
                            List[SEALDATE] = line2[1].lstrip(r' SEALDTealate')
                        else:
                            List[SEALDATE] = line[1][0].lstrip(r' SEALDTealate')
                        overwrite[SEALDATE] = 1
                        EID = 0
                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
                        if overwrite[a] == 0:
                            for res in range(len(decode_result)):
                                if wrote[res] != 'wrote' and decode_result[res] != '':
                                    print(decode_result[res])
                                    List[a] = decode_result[res]
                                    overwrite[a] = 1
                                    wrote[res] = 'wrote'
                                    break
                writer.writerow(List)

        # AKOUSTIS
        elif (Comp == 5):
            result_path = './result_dir/Company_OCR/AKOUSTIS_csv.csv'
            with open(result_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                Company, Part, LOT, MFG, DTE, QTY = 1, 2, 3, 4, 5,6 # 哪一項放在第幾格
                List = ['-', '-', '-', '-', '-', '-', '-']
                EID = 0  # 換行用
                s = str(path)
                List[0] = s.strip("/content/LABEL/")  # 第一格我放圖片名稱
                overwrite = [0, 0, 0, 0, 0, 0, 0]  # 看要輸入的格子裡面是不是已經有資料時用
                index = 0
                for line in result:
                    line2 = line[1][0]
                    # print(line2)

                    # Company
                    if 'AKOUSTIS' in line[1][0] and overwrite[Company] == 0:  # 那行有公司名
                        List[Company] = 'AKOUSTIS'  # 填公司名
                        overwrite[Company] = 1  # 填了

                        EID = 0  # 不用換行

                    # Part#
                    elif ('Part' in line[1][0]) and overwrite[Part] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[Part] = line2[1].lstrip(' ')  # 那行有被分割過(有冒號) 填第2個資料 lstrip(刪前面空格)
                        overwrite[Part] = 1  # 填了

                        EID = 0  # 不用換行

                    # LOT#
                    elif ('LOT' in line[1][0] or 'P/N' in line[1][0]) and overwrite[LOT] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[LOT] = line2[1].lstrip(' ')
                        overwrite[LOT] = 1

                        EID = 0

                    # MFG#
                    elif 'MFG' in line[1][0] and overwrite[MFG] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[MFG] = line2[1].lstrip(' ')
                        overwrite[MFG] = 1

                        EID = 0

                    # DTE
                    elif 'DTE' in line[1][0] and overwrite[DTE] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[DTE] = line2[1].lstrip(' ')
                        overwrite[DTE] = 1

                        EID = 0

                    # QTY
                    elif 'QTY' in line[1][0] and overwrite[QTY] == 0:
                        if ':' in line[1][0] :
                            line2 = line2.split(':')  # line[1][0]是偵測到整行的 line2有用冒號分割
                        elif'#'in line[1][0] :
                            line2 = line2.split('#')  # line[1][0]是偵測到整行的 line2有用#分割
                        if len(line2) > 1: List[QTY] = line2[1].lstrip(' ')
                        overwrite[QTY] = 1

                        EID = 0

                #######################################
                overwrite[0] = 1
                if decode_result != []:
                    wrote = decode_result
                    for a in range(len(overwrite)):
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
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()
    cv2.destroyAllWindows()
    print("done")

def photo_obj_detection_2(model_path,GPU_ratio=0.8):





    # matplot show函式
    def mat_show(image):
        plt.imshow(image)
        plt.show()

    def compare(str1, str2):
        tmp1 = str1.replace(" ", "")
        tmp2 = str2.replace(" ", "")

        if tmp1 in tmp2 or tmp2 in tmp1:
            return True
        else:
            return False

    # paddleOCR辨識
    ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
    img_path = "./Input_dir/ALL_company/THALES(14).jpg"  # 選擇用哪張照片

    result = ocr.ocr(img_path, cls=False)

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
            res1 = res[1][0]
            res1 = res1.split(spilt_tmp)
            if write == 1:
                write = 0
                List[pre] = res1[0]
                index += 1
            if compare(col_tmp, res[1][0]):
                if len(res1[1]) > 1:
                    List[index] = res1[1]
                    index += 1
                else:
                    write = 1
                    pre = index

    print(List)
    #####################################################
    # ----release
    # f.close()
    # fc.close()
    # yolo_v4.sess.close()

    cv2.destroyAllWindows()
    print("done")

def cross_photo_obj_detection(model_path, GPU_ratio=0.8):
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

    # paddleOCR辨識
    ocr = PaddleOCR(lang='en')
    result_top = ocr.ocr(img_top, cls=False)
    result_side = ocr.ocr(img_side, cls=False)
    result = result_top+result_side

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
        f.write(res[1][0] + '\n')
        print(res[1][0])

    # 印出Barcode/QRCode內容
    print("Barcode/QRCode Part:\n\n")
    if decode_result != []:
        for res in decode_result:
            fc.write(res + '\n')
            print(res)
    else:
        print("Decode Fail")

    #####################################################
    # ----release
    f.close()
    fc.close()
    yolo_v4.sess.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r".\yolov4-obj_best_416.ckpt.meta"
    # model_path = r"C:\Users\shiii\YOLO_v4-master\yolov4_416.ckpt.meta"
    GPU_ratio = 0.8
    real_time_obj_detection(model_path,GPU_ratio=GPU_ratio)
    # photo_obj_detection(model_path,GPU_ratio=GPU_ratio)

    # photo_obj_detection_2(model_path,GPU_ratio=GPU_ratio)
    # cross_photo_obj_detection(model_path,GPU_ratio=GPU_ratio)