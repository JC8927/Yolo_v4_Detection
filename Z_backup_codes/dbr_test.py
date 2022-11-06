# -*- coding: UTF-8 -*-
from dbr import *
import cv2
import matplotlib
import matplotlib.pyplot as plt
# 導入需要套件
import cv2
import pyzbar.pyzbar as pyzbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 建立BarcodeReader
BarcodeReader.init_license("t0076oQAAADLDNLLexPCL5vfn2vtVNtjVvYQzSHAmkcuhnLZhwoyd50yzV5xlNT6PYgMhdBsXn72R4cNUcOLv82zt0jv+NFJb2RQn/4Yi6Q==")
reader = BarcodeReader()

# Barcode reader setting
settings = reader.get_runtime_settings()
settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_POSTALCODE | EnumBarcodeFormat_2.BF2_DOTCODE
settings.excepted_barcodes_count = 35
reader.update_runtime_settings(settings)


# 讀取barcode圖片
img_name = r"C:\Users\shiii\Yolo_v4_Detection\Input_dir\Test_img\12.jpg"
# 印出barcode圖片
print(img_name+":")
img = mpimg.imread(img_name)
imgplot = plt.imshow(img)
plt.show()

try:
    image = img_name
    text_results = reader.decode_file(image)
    if text_results != None:
        for text_result in text_results:
#             print("Barcode Format : " + text_result.barcode_format_string)
            print("Barcode Text : " + text_result.barcode_text)
            if len(text_result.barcode_format_string) == 0:
                pass
#                 print("Barcode Format : " + text_result.barcode_format_string_2)
#         else:
#             print("Barcode Format : " + text_result.barcode_format_string)
#             print("Barcode Text : " + text_result.barcode_text)
except BarcodeReaderError as bre:
    print(bre)