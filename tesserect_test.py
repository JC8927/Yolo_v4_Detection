import cv2,time
import numpy
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
tesseract_result_path = './result_dir/tesseract_result_txt.txt'

img_path = './result_dir/result_pic_yolo_crop.jpg'
Tesserect_result = pytesseract.image_to_string(img_path, lang="chi_tra+eng")

img = Image.open(".\Input_dir\ALL_company\EDOM(3).jpg")
string = pytesseract.image_to_string(img, lang="chi_tra+eng")

tesseract_f = open(tesseract_result_path, 'w')

for res in Tesserect_result.split('\n'):
    tesseract_f.write(res + '\n')
    print(res)
