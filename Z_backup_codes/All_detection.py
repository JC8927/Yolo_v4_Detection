import cv2,time
import numpy
import numpy as np
import tensorflow
from src.YOLO import YOLO
from src.Feature_parse_tf import get_predict_result
from utils import tools

from paddleocr import PaddleOCR,draw_ocr
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import copy

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
        name_file = "../barcode.names"

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
        # 保留沒有標記的圖片
        # ret_1, img = cap.read()
        # ----get image
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
                cv2.imwrite('../result_dir/result_pic_orig.jpg', pic)
                # input("Please press the Enter key to proceed")
                # 儲存yolo辨識照片
                cv2.imwrite('../result_dir/result_pic_yolo.jpg', yolo_img)
                # input("Please press the Enter key to proceed")

                # 將yolo找到的code部分刪掉
                # 讀取yolo找到的座標
                with open(r'../result_dir/yolo_box.txt', 'r') as f:
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
                cv2.imwrite('../result_dir/result_pic_yolo_crop.jpg', img)
                print("***************************")
                #####################################################
                # paddleOCR辨識
                ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
                img_path = '../result_dir/result_pic_yolo_crop.jpg'
                result = ocr.ocr(img_path, cls=False)
                decode_result = pyz_decoded_str
                # 匯出辨識結果(txt)
                result_path = '../result_dir/result_txt.txt'
                decode_result_path = '../result_dir/decode_result_txt.txt'
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
    cv2.imwrite('../result_dir/result_pic_orig.jpg', pic)
    # input("Please press the Enter key to proceed")
    # 儲存yolo辨識照片
    cv2.imwrite('../result_dir/result_pic_yolo.jpg', yolo_img)
    # input("Please press the Enter key to proceed")

    # 將yolo找到的code部分刪掉
    # 讀取yolo找到的座標
    with open(r'../result_dir/yolo_box.txt', 'r') as f:
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
    cv2.imwrite('../result_dir/result_pic_yolo_crop.jpg', img)

    #####################################################
    # paddleOCR辨識
    ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
    img_path = '../result_dir/result_pic_orig.jpg'
    result = ocr.ocr(img_path, cls=False)
    decode_result = pyz_decoded_str
    # 匯出辨識結果(txt)
    result_path = '../result_dir/result_txt.txt'
    decode_result_path = '../result_dir/decode_result_txt.txt'
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
    # ----release
    f.close()
    fc.close()
    yolo_v4.sess.close()
    cap.release()


    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

def photo_obj_detection(model_path,GPU_ratio=0.8):
    #----var


    #----YOLO v4 init
    yolo_v4 = Yolo_v4(model_path,GPU_ratio=GPU_ratio)
    img = cv2.imread('../Input_dir/ALL_company/THALES(11).jpg')

    pic = numpy.array(img)
    # ----YOLO v4 detection
    yolo_img ,pyz_decoded_str= yolo_v4.detection(img)

    # #####################################################
    # # 儲存原始照片
    # cv2.imwrite('./result_dir/result_pic_orig.jpg', pic)
    # # input("Please press the Enter key to proceed")
    # # 儲存yolo辨識照片
    # cv2.imwrite('./result_dir/result_pic_yolo.jpg', yolo_img)
    # # input("Please press the Enter key to proceed")
    # #####################################################
    # paddleOCR辨識
    ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory
    img_path = '../Input_dir/ALL_company/THALES(11).jpg'
    result = ocr.ocr('./input_dir/ALL_company/THALES(11).jpg', cls=False)
    decode_result=pyz_decoded_str
    # 匯出辨識結果(txt)
    result_path = '../result_dir/result_txt.txt'
    decode_result_path = '../result_dir/decode_result_txt.txt'
    f = open(result_path, 'w')
    fc=open(decode_result_path, 'w')
    # 印出字元與位置
    '''for line in result:
        f.write(res[1]+'\n')
        print(line)'''
    # 印出字元
    print("Text Part:\n")
    for res in result:
        f.write(res[1][0]+'\n')
        print(res[1][0])
    # 印出Barcode/QRCode內容
    print("Barcode/QRCode Part:\n\n")
    if decode_result!=[]:
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
    model_path = r"../yolov4-obj_best_416.ckpt.meta"
    # model_path = r"C:\Users\shiii\YOLO_v4-master\yolov4_416.ckpt.meta"
    GPU_ratio = 0.8
    real_time_obj_detection(model_path,GPU_ratio=GPU_ratio)
    # photo_obj_detection(model_path,GPU_ratio=GPU_ratio)
