from typing import List,Union
import difflib
import numpy as np
import json
import os
import copy
import cv2
from fuzzywuzzy import fuzz
def check_result_list(result_list):
    c_result_list = sorted(result_list,key= lambda d:d['col_name'])
    max_col_num = 0
    col_num = 0
    col_name = None
    for result in c_result_list:
        if  col_name == None or col_name == result['col_name']:
            col_name = result['col_name']
            col_num = col_num + 1
        else:
            if max_col_num==0:
                max_col_num = col_num
                col_name = result['col_name']
                col_num = 1
                continue
            col_name = result['col_name']
            col_num = 1
    col_num = 0
    col_name = None
    for result in c_result_list:
        if  col_name == None:
            diction = copy.deepcopy(result)
            col_name = result['col_name']
            col_num = col_num + 1
        elif col_name == result['col_name']:
            col_name = result['col_name']
            col_num = col_num + 1
        else:
            if col_num<max_col_num:
                print(col_name+"有缺漏")
                #代表整張label只有一個此類型col_data 預設為整張適用
                if col_num == 1:
                    for i in range(1,max_col_num):
                        diction['col_id'] = i
                        result_list.append(diction)
            diction = copy.deepcopy(result)
            col_name = result['col_name']
            col_num = 1
    return result_list

def img_resize(img):
    height,width=img.shape[0],img.shape[1]
    renew_length=900#自定義最長邊愈拉長至多長
    if width/height>=1:#(width>height) 拉長width至愈調整尺寸
        img_new=cv2.resize(img,(renew_length,int(height*renew_length/width)))
    else:#(height>width) 拉長height至愈調整尺寸
        img_new=cv2.resize(img,(int(width*renew_length/height),renew_length))
    return img_new

def result_grouping(result_list,img):

    idx_img = img.copy()
    for result in result_list:
        bounding_poly=result['bounding_poly']
        min_x=None
        min_y=None
        max_x=None
        max_y=None

        for axis in bounding_poly:
            cur_x=axis[0]
            cur_y=axis[1]
            if min_x is None or min_x>cur_x:
                min_x=cur_x
            if max_x is None or max_x<cur_x:
                max_x=cur_x
            if min_y is None or min_y>cur_y:
                min_y=cur_y
            if max_y is None or max_y<cur_y:
                max_y=cur_y

        cv2.rectangle(idx_img, (min_x, min_y), (max_x, max_y), (0,0,255),thickness=3)
    idx_img=img_resize(idx_img)
    cv2.namedWindow("show_col", cv2.WINDOW_NORMAL)
    crop_y = int(1*idx_img.shape[0])
    crop_x = int(1*idx_img.shape[1])
    cv2.resizeWindow("show_col",crop_x,crop_y)
    cv2.imshow("show_col",idx_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows("show_col")
    
def draw_final_pic(combined_result,img_path):
    img=cv2.imread(img_path)
    idx_img = img.copy()
    color_list=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for result in combined_result:
        now_label_id=result['label_id']
        barcode_bounding_poly=result['barcode_bounding_poly']
        ocr_bounding_poly=result['ocr_bounding_poly']
        min_x=None
        min_y=None
        max_x=None
        max_y=None

        for axis in ocr_bounding_poly:
            cur_x=axis[0]
            cur_y=axis[1]
            if min_x is None or min_x>cur_x:
                min_x=cur_x
            if max_x is None or max_x<cur_x:
                max_x=cur_x
            if min_y is None or min_y>cur_y:
                min_y=cur_y
            if max_y is None or max_y<cur_y:
                max_y=cur_y
        cv2.rectangle(idx_img, (min_x, min_y), (max_x, max_y), color_list[now_label_id],thickness=3)
        # idx_img_test=img_resize(idx_img.copy())
        # cv2.namedWindow("show_col", cv2.WINDOW_NORMAL)
        # crop_y = int(1*idx_img.shape[0])
        # crop_x = int(1*idx_img.shape[1])
        # cv2.resizeWindow("show_col",crop_x,crop_y)
        # cv2.imshow("show_col",idx_img)
        # cv2.waitKey(0)
        min_x=None
        min_y=None
        max_x=None
        max_y=None

        for axis in barcode_bounding_poly:
            cur_x=axis[0]
            cur_y=axis[1]
            if min_x is None or min_x>cur_x:
                min_x=cur_x
            if max_x is None or max_x<cur_x:
                max_x=cur_x
            if min_y is None or min_y>cur_y:
                min_y=cur_y
            if max_y is None or max_y<cur_y:
                max_y=cur_y

        cv2.rectangle(idx_img, (min_x, min_y), (max_x, max_y), color_list[now_label_id],thickness=3)
    idx_img=img_resize(idx_img)
    cv2.namedWindow("show_col", cv2.WINDOW_NORMAL)
    crop_y = int(1*idx_img.shape[0])
    crop_x = int(1*idx_img.shape[1])
    cv2.resizeWindow("show_col",crop_x,crop_y)
    cv2.imshow("show_col",idx_img)
    cv2.waitKey(1)


#觀察:(XX) XX為用於qrcode中的標示
def data_preprocess(results)-> List[dict]:#傳入圖片ocr辨識結果  
    imformation_list=[]
    record_list=['QTY','LOT','DATE','PN','COO'] #紀錄事項會從config載入
    record_dict={'QTY':['OTY',"Q", "<QTY>", "Box Qty", "QTY", "QTY’", "QUANTITY", "Qty.", "Q’ ty", "Q’ty", "TOTALQTY", "Total Qty", "Unit Q’ty"],'LOT':["Bulk ID","1T", "<LOT>", "L/C", "L/N", "LN", "LOT", "LOT NO", "LOT NUMBER", "LOT NUMBERS", "LOT#", "LOTPO", "Lot Code", "Lot ID", "LOTNO", "Lot No.", "MLOT"],'DATE':["D","Trace code1","Trace codes","9D", "Assembly Date Code", "D/C", "DATE", "DATE CODE", "DATECODE", "DC", "DCODE", "DTE", "Seal Date"],'PN':["Type","1P", "P", "<P/N>", "CPN", "MPN", "P/N", "P/N Name", "PART", "PART ID", "PART NO", "PART NUMBER", "PN", "PROD ID", "PROD NO", "PRODUCT ID", "Part No.", "PartNo", "SPN"],'COO':["4L","Assembled In", "Assembly Location", "C.O.O.", "COO", "COUNTRY OF ORIGIN", "MADE IN", "Origin of"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    for col_list_name in record_dict.keys():
        col_list=record_dict[col_list_name]
        col_list=[col_name.upper() for col_name in col_list]
        for i,col_name in enumerate(col_list):
            for j,com_col_name in enumerate(col_list):
                if col_name==com_col_name and i!=j:
                    print("重複")
    #處理data
    for j in range(len(results)):
        match_col_name_list=[]
        location=results[j][0] #文字方框位置
        location_mid_x=int((location[0][0]+location[1][0])/2) #找出方框x中點
        location_mid_y=int((location[1][1]+location[2][1])/2) #找出方框y中點
        whole_text=results[j][1][0].upper()#找出偵測文字
        #text=text.replace(" ","")
        score=results[j][1][1]#找出文字分數
        pre_text = None
        after_text = None
        match_text = None
        if whole_text.find(' ') != -1:#可能要再改
            split_flag=True
            text_list=whole_text.split(" ")
            for i,text in enumerate(text_list):
                first_flag = False
                match_flag = False
                after_text = None
                cur_text=text.upper()

                for col_list_name in record_dict.keys():

                    col_list=record_dict[col_list_name]
                    col_list=[col_name.upper() for col_name in col_list]
                    for col_name in col_list:

                        if col_name.find(' ') != -1:
                            #col_name 中間有空格
                            col_text_list=col_name.split(" ")
                            for k,text in enumerate(col_text_list):
                                cur_col_text = text.upper()
                                if cur_col_text ==  cur_text:
                                    if (i+1) != len(text_list):
                                        after_text = text_list[i+1]
                                    if k == 0:
                                        first_flag = True
                                    match_flag = True
                                    diction={'cur_text':cur_col_text,'after_text':after_text,'col_list_name':col_list_name,'col_name':col_name,'text_order':i}
                                    match_col_name_list.append(diction)
                                #對應到col_name第一個就不符合 直接跳出
                                # if first_flag == False:
                                #     break
                        else:
                            if col_name == cur_text:
                                if (i+1) != len(text_list):
                                    after_text = text_list[i+1]
                                match_flag = True
                                first_flag = True
                                diction={'cur_text':cur_text,'after_text':after_text,'col_list_name':col_list_name,'col_name':col_name,'text_order':i}
                                match_col_name_list.append(diction)

                if first_flag == False or first_flag ==True:
                    diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':False}
                    imformation_list.append(diction)

            if len(match_col_name_list) != 0:
                match_col_name_list=sorted(match_col_name_list,key=lambda d:d['text_order'])
                temp_imformation_list=[]
                for diction in match_col_name_list:
                    col_list_name = diction['col_list_name']
                    now_col_name = diction['col_name'].upper()
                    col_list=record_dict[col_list_name]
                    cur_text = diction['cur_text']
                    after_text = diction['after_text']
                    start_order = diction['text_order']
                    end_order = None
                    combined_text = None
                    #now_col_name 有空格
                    if after_text != None and now_col_name.find(' ') != -1:
                        combined_text = cur_text+" "+after_text
                        for after_text_diction in match_col_name_list:
                            after_cur_text = after_text_diction['cur_text']
                            if after_text == after_cur_text and now_col_name == after_text_diction['col_name']:
                                cur_text = after_text_diction['cur_text']
                                after_text = after_text_diction['after_text']
                                end_order = after_text_diction['text_order']
                                if after_text != None and len(combined_text.split()) != len(now_col_name.split()):
                                    combined_text = combined_text+" "+str(after_text)
                        if combined_text == now_col_name:
                            diction={'x':location_mid_x,'y':location_mid_y,'text':combined_text,'bounding_poly':location,'score':score,'col_name_flag':True,'start_order':start_order,'end_order':end_order}
                            temp_imformation_list.append(diction)
                    else:
                        if after_text != None and now_col_name == cur_text:
                            combined_text = cur_text+" "+after_text
                            diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':True,'start_order':start_order,'end_order':end_order}
                            temp_imformation_list.append(diction)
                        elif now_col_name == cur_text:
                            diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':True,'start_order':start_order,'end_order':end_order}
                            temp_imformation_list.append(diction)
                while(len(temp_imformation_list)!=0):
                    for diction in temp_imformation_list:
                        break_flag = False
                        end_order = diction['end_order']
                        if end_order == None:
                            end_order = -1
                        start_order = diction['start_order']
                        for compared_diction in temp_imformation_list:
                            compared_start_order = compared_diction['start_order']
                            compared_end_order = compared_diction['end_order']
                            if compared_end_order == None:
                                compared_end_order = -1
                            if diction == compared_diction:
                                continue
                            if start_order == compared_start_order:
                                if end_order ==-1:
                                    temp_imformation_list.remove(diction)
                                    break_flag = True
                                    break
                                elif compared_end_order == -1:
                                    temp_imformation_list.remove(compared_diction)
                                    break_flag = True
                                    break
                                else:
                                    diff_end_order = end_order-compared_end_order
                                    if diff_end_order>0:
                                        temp_imformation_list.remove(compared_diction)
                                        break_flag = True
                                        break
                                    else:
                                        temp_imformation_list.remove(diction)
                                        break_flag = True
                                        break
                            elif compared_start_order<=end_order and compared_start_order>=start_order:
                                temp_imformation_list.remove(compared_diction)
                                break_flag = True
                                break
                        if break_flag:
                            break
                        else:
                            imformation_list.append(diction)
                            temp_imformation_list.remove(diction)

                    # for col_name in col_list:
                    #     if combined_text == col_name:

        else:
            cur_text = whole_text
            match_flag = False
            for col_list_name in record_dict.keys():
                col_list=record_dict[col_list_name]
                col_list=[col_name.upper() for col_name in col_list]
                for col_name in col_list:
                    if col_name == cur_text :
                        match_flag = True
                        diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':True}
                        imformation_list.append(diction)
            if match_flag == False:
                diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':False}
                imformation_list.append(diction)


    imformation_list=sorted(imformation_list,key=lambda d:d['y'])#由y軸座標排序
    new_imformation_list=[]
    #檢查重複
    for j in range(len(results)):
        location=results[j][0] #文字方框位置
        location_mid_x=int((location[0][0]+location[1][0])/2) #找出方框x中點
        location_mid_y=int((location[1][1]+location[2][1])/2) #找出方框y中點
        #找出相對應的imformation
        compared_imformation_list=[]
        compared_imformation_text_list=[]
        for imformation in imformation_list:
            if imformation['bounding_poly'] == location:
                compared_imformation_text_list.append(imformation['text'])
                compared_imformation_list.append(imformation)
        #切字到最小單位
        whole_text=results[j][1][0].upper()
        if whole_text.find(' ') != -1:#可能要再改
            split_flag=True
            text_list=whole_text.split(" ")
            #-1表未改變 0表一般字 1表col_data 單一 2 表col_data 連續
            text_list_property=[-1 for i in range(len(text_list))]
        else:
            text_list=[whole_text]
            text_list_property=[-1]
        #先check 最長
        for imformation in compared_imformation_list:
            if imformation['col_name_flag'] == True:
                #是 col_data 且 不只一個字
                if imformation['text'].find(' ')!= -1:
                    imformation_text_list=imformation['text'].split(" ")
                    for i,text in enumerate(text_list):
                        if text == imformation_text_list[0]:
                            flag= True
                            for j in range(len(imformation_text_list)):
                                if imformation_text_list[j] == text_list[i+j] and text_list_property[i+j]==-1:
                                    continue
                                else:
                                    flag = False
                            if flag ==True:
                                for j in range(len(imformation_text_list)):
                                    text_list_property[i+j]=2
        #check 單一 col_data
        for imformation in compared_imformation_list:
            if imformation['col_name_flag'] == True:
                #是 col_data 且 不只一個字
                if imformation['text'].find(' ')== -1:
                    imformation_text = imformation['text']
                    for i,text in enumerate(text_list):
                        if text == imformation_text and text_list_property[i]==-1:
                            text_list_property[i]=1
        #check 非 col_data
        for imformation in compared_imformation_list:
            if imformation['col_name_flag'] == False:
                #是 col_data 且 不只一個字
                if imformation['text'].find(' ')== -1:
                    imformation_text = imformation['text']
                    for i,text in enumerate(text_list):
                        if text == imformation_text and text_list_property[i]==-1:
                            text_list_property[i]=0
        #print(text_list_property)
        skip_num = 0
        for i in range(len(text_list_property)):
            if skip_num!=0:
                skip_num = skip_num-1
                continue
            if text_list_property[i] == 0:
                cur_text = text_list[i]
                score = 1
                diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':False}
                new_imformation_list.append(diction)
            elif text_list_property[i] == 1:
                cur_text = text_list[i]
                score = 1
                diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':True}
                new_imformation_list.append(diction)
            elif text_list_property[i] == 2:
                cur_text =text_list[i]
                while text_list_property[i] == 2 :
                    i=i+1
                    if i == len(text_list_property):
                        break
                    if text_list_property[i] == 2:
                        skip_num=skip_num+1
                        cur_text = cur_text+" "+text_list[i]
                    else:
                        i=i-1
                        break

                score = 1
                diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score,'col_name_flag':True}
                new_imformation_list.append(diction)
    #print(new_imformation_list)
    new_imformation_list=sorted(new_imformation_list,key=lambda d:d['y'])
                    

    return new_imformation_list

def barcode_data_preprocess(codes)-> List[dict]:#傳入barcode結果 
    diction_list=[]
    for code in codes:
        text=code.data.decode('utf-8')
        pts_rect=np.array(code.rect,np.int32)
        loc_mid_x=int(pts_rect[0]+pts_rect[2]/2)
        loc_mid_y=int(pts_rect[1]+pts_rect[3]/2)
        diction={"text":text,"x":loc_mid_x,"y":loc_mid_y}
        diction_list.append(diction)
    diction_list=sorted(diction_list,key=lambda d:d['y'])#由y軸座標排序
    return diction_list

def search_col_data_idx(idx,number,imformation_list):
    possible_col_data_idx_list=[]
    pre_idx=None
    next_idx=None

    if idx != 0:
        pre_idx=idx-1
    if idx !=(len(imformation_list)-1):
        next_idx=idx+1
    next_flag=True
    pre_flag=True
    while len(possible_col_data_idx_list)<number:

        if next_idx != None:
            while(imformation_list[next_idx]['x']<imformation_list[idx]['x']) and next_idx < (len(imformation_list)-1):
                next_idx=next_idx+1
            if next_flag:
                if next_idx != (len(imformation_list)-1) :
                    possible_col_data_idx_list.append(next_idx-idx)
                    next_idx=next_idx+1
                elif next_idx == (len(imformation_list)-1) and imformation_list[next_idx]['x']>imformation_list[idx]['x'] :
                    possible_col_data_idx_list.append(next_idx-idx)
                    next_idx=next_idx+1
                else:
                    next_idx=next_idx+1
            if next_idx>(len(imformation_list)-1):
                next_flag=False
                next_idx=next_idx-1
                
            
        if pre_idx != None:
            while(imformation_list[pre_idx]['x']<imformation_list[idx]['x']) and pre_idx>0:
                pre_idx=pre_idx-1
            if pre_flag:
                if pre_idx != 0 :
                    possible_col_data_idx_list.append(pre_idx-idx)
                    pre_idx=pre_idx-1
                elif pre_idx == 0 and imformation_list[pre_idx]['x']>imformation_list[idx]['x'] :
                    possible_col_data_idx_list.append(pre_idx-idx)
                    pre_idx=pre_idx-1
                else:
                    pre_idx=pre_idx-1
            if pre_idx<0:
                pre_flag=False
                pre_idx=pre_idx+1
        
        if next_flag==False and pre_flag == False:
            break

    return possible_col_data_idx_list

def compare_col_data(imformation_list,col_name,config_list,config,col_data_idx_list,idx,idx_col_data,col_name_x,col_name_y,pass_num):
    col_data_list=[]
    col_data_correct_flag = False
    #col_data查找範圍
    now_data_idx_list = None
    col_data = None
    saved_flag=False
    i = None
    check =None
    detect_mode = "0"
    mode_check = "0"
    data_num = 0
    if config != None:
        data_num = int(config['text_num'])
        now_data_idx_list=config['col_data_idx']
        detect_mode = config['detect_mode']
        col_name = config['col']
        config_text_len_list = config['text_length']
        config_text_type_list = config['text_type']
        config_col_data_idx_list = config['col_data_idx']
    #找 col_data_list
    
    if now_data_idx_list != None:
        #左右模式
        if detect_mode =="0":
            for data_idx in col_data_idx_list:
                if data_idx+idx<len(imformation_list) and data_idx+idx>-1:
                    col_data = imformation_list[idx+data_idx]['text']
                    diff_y = imformation_list[idx+data_idx]['y']-imformation_list[idx]['y']
                    diff_x = imformation_list[idx+data_idx]['x']-imformation_list[idx]['x']
                    col_name_flag = imformation_list[idx+data_idx]['col_name_flag']
                    if diff_x >=0:
                        diction={'col_data':col_data,'diff_y':abs(diff_y),'diff_x':diff_x,'idx':idx+data_idx,'col_name_flag':col_name_flag}
                        col_data_list.append(diction)
                else:
                    continue
        #上下模式
        elif detect_mode == "1":
            for data_idx in col_data_idx_list:
                if data_idx+idx<len(imformation_list):
                    col_data = imformation_list[idx+data_idx]['text']
                    diff_y = imformation_list[idx+data_idx]['y']-imformation_list[idx]['y']
                    diff_x = imformation_list[idx+data_idx]['x']-imformation_list[idx]['x']
                    col_name_flag = imformation_list[idx+data_idx]['col_name_flag']
                    if diff_y>=0:
                        diction={'col_data':col_data,'diff_y':diff_y,'diff_x':abs(diff_x),'idx':idx+data_idx,'col_name_flag':col_name_flag}
                        col_data_list.append(diction)
                else:
                    continue
    else:
        num = 15
        print("請輸入辨識模式 上下順序請輸入:1 預設為左右順序 直接Enter")
        mode_check = input()
        detect_mode = mode_check
        col_data_idx_list = [i for i in range(-num,num,1)]
        if mode_check == "1":
            for data_idx in col_data_idx_list:
                if data_idx+idx<len(imformation_list) and data_idx+idx>-1:
                    col_data = imformation_list[idx+data_idx]['text']
                    diff_y = imformation_list[idx+data_idx]['y']-imformation_list[idx]['y']
                    diff_x = imformation_list[idx+data_idx]['x']-imformation_list[idx]['x']
                    col_name_flag = imformation_list[idx+data_idx]['col_name_flag']
                    if config != None:
                        now_data_idx=config['col_data_idx']
                    if diff_y >=0:
                        diction={'col_data':col_data,'diff_y':diff_y,'diff_x':abs(diff_x),'idx':idx+data_idx,'col_name_flag':col_name_flag}
                        col_data_list.append(diction)
                else:
                    continue
        else:
            mode_check = "0"
            detect_mode = "0"
            for data_idx in col_data_idx_list:
                if data_idx+idx<len(imformation_list) and data_idx+idx>-1:
                    col_data = imformation_list[idx+data_idx]['text']
                    diff_y = imformation_list[idx+data_idx]['y']-imformation_list[idx]['y']
                    diff_x = imformation_list[idx+data_idx]['x']-imformation_list[idx]['x']
                    col_name_flag = imformation_list[idx+data_idx]['col_name_flag']
                    if config != None:
                        now_data_idx=config['col_data_idx']
                    if diff_x >=0:
                        diction={'col_data':col_data,'diff_y':abs(diff_y),'diff_x':diff_x,'idx':idx+data_idx,'col_name_flag':col_name_flag}
                        col_data_list.append(diction)
                else:
                    continue
    if detect_mode == "1":

        col_data_list = sorted(col_data_list,key=lambda d:d['diff_y'])
        col_data_list = sorted(col_data_list,key=lambda d:d['diff_x'])
    else:
        col_data_list = sorted(col_data_list,key=lambda d:d['diff_x'])
        col_data_list = sorted(col_data_list,key=lambda d:d['diff_y'])
    
    #改成可調整紀錄 data數量
    text_len_list = []
    text_type_list=[]
    text_num = 0
    recorded_data_list = []
    recorded_data_idx_list = []
    col_data_correct_flag = False

    for i,col_dict in enumerate(col_data_list):
        col_data = col_dict['col_data']
        col_data_idx = col_dict['idx']-idx
        col_name_flag = col_dict['col_name_flag']
        data = col_name+":"+col_data
        if config == None:
            if col_name_flag == True:
                continue
            print(data)
            print("請問是否符合"+col_name+"格式?如果符合請輸入:T 如果紀錄完畢請按:O")
            check=input()
            if check.upper()=="T":

                text_len=len(col_data) #儲存長度參數
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type='int'
                text_type_dict={"LOT":"str",'PN':'str','COO':'str','DATE':'str'}
                for name in text_type_dict.keys():
                    if name == col_name:
                        text_type = text_type_dict[name]
                text_type_list.append(text_type)
                text_len_list.append(text_len)
                recorded_data_list.append(col_data)
                recorded_data_idx_list.append(col_data_idx)
                text_num = text_num+1
                col_data_correct_flag=True

                if i == (len(col_data_list)-1):
                    saved_imformation_list = copy.deepcopy(imformation_list)
                    saved_imformation_list = sorted(saved_imformation_list,key = lambda d:d['x'])
                    if mode_check == "1":
                        for i,imformation in enumerate(saved_imformation_list) :
                            if imformation == imformation_list[idx]:
                                idx = i
                                break
                        config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                        config=config_diction
                        config_list.append(config_diction)
                        saved_flag = True
                    elif mode_check == "0":
                        config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                        config=config_diction
                        config_list.append(config_diction)
                        saved_flag = True
                    return recorded_data_list,recorded_data_idx_list


            elif check.upper()=="O" and text_num != 0:
                saved_imformation_list = copy.deepcopy(imformation_list)
                saved_imformation_list = sorted(saved_imformation_list,key = lambda d:d['x'])
                if detect_mode == "1":
                    for i,imformation in enumerate(saved_imformation_list) :
                        if imformation == imformation_list[idx]:
                            idx = i
                            break
                    config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                    config=config_diction
                    config_list.append(config_diction)
                    saved_flag = True
                elif detect_mode == "0":
                    config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                    config=config_diction
                    config_list.append(config_diction)
                    saved_flag = True
                return recorded_data_list,recorded_data_idx_list
            elif check.upper()=="O" and text_num == 0:
                break
            elif i == (len(col_data_list)-1) and text_num != 0:
                if i == (len(col_data_list)-1):
                    saved_imformation_list = copy.deepcopy(imformation_list)
                    saved_imformation_list = sorted(saved_imformation_list,key = lambda d:d['x'])
                    if mode_check == "1":
                        for i,imformation in enumerate(saved_imformation_list) :
                            if imformation == imformation_list[idx]:
                                idx = i
                                break
                        config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                        config=config_diction
                        config_list.append(config_diction)
                    elif mode_check == "0":
                        config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
                        config=config_diction
                        config_list.append(config_diction)
                    return recorded_data_list,recorded_data_idx_list

            
            

        else:


            text_len=len(col_data)
            text_type=type(col_data).__name__

            if col_name_flag:
                continue
            if text_type=='str':
                if col_data.isdigit():
                    text_type='int'
            record_list=['QTY','LOT','DATE','PN','COO']
            text_type_dict={"LOT":"str",'PN':'str','COO':'str','DATE':'str'}
            for name in text_type_dict.keys():
                if name == col_name:
                    text_type = text_type_dict[name]

            # if col_name == 'COO':#國家
            #     country_list=['TW','CN','CHINA','MALAYSIA','TH'] #加入國家 作為檢索資料
            #     for country in country_list:
            #         if country.upper() == col_data.upper():
            #             col_data_correct_flag = True
            #             recorded_data_list.append(country.upper())
            #             break
            #     if col_data_correct_flag:
            #         break
            #     continue
            detected_flag = False
            detected_idx = None
            for j,config_text_type in enumerate(config_text_type_list):
                if text_type == config_text_type:

                    if col_name == 'COO':#國家
                        
                        country_list=['TW','CN','CHINA','MALAYSIA','TH']
                        for country in country_list:
                            if country.upper() == col_data.upper():
                                col_data_correct_flag = True
                                detected_flag = True
                                detected_idx = j
                                recorded_data_list.append(country.upper())
                                recorded_data_idx_list.append(col_data_idx)
                                break
                        if col_data_correct_flag==True:
                            break
                        else:
                            continue

                    if text_type=='str':
                        if text_len != config_text_len_list[j]:
                            continue
                        else:
                            col_data_correct_flag = True
                            detected_flag = True
                            detected_idx = j
                            recorded_data_idx_list.append(col_data_idx)
                            recorded_data_list.append(col_data)
                            break
                    elif text_type == 'int':
                        col_data_correct_flag = True
                        detected_flag = True
                        detected_idx = j
                        recorded_data_list.append(col_data)
                        recorded_data_idx_list.append(col_data_idx)
                        break
            if detected_flag:
                config_text_len_list.pop(detected_idx)
                config_text_type_list.pop(detected_idx)
                if len(config_text_len_list)==0:
                    break
    if col_data_correct_flag == False:
        recorded_data_list=None
        recorded_data_idx_list=None
    if saved_flag ==False:
        saved_imformation_list = copy.deepcopy(imformation_list)
        saved_imformation_list = sorted(saved_imformation_list,key = lambda d:d['x'])
        if detect_mode == "1":
            for i,imformation in enumerate(saved_imformation_list) :
                if imformation == imformation_list[idx]:
                    idx = i
                    break
            config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
            config=config_diction
            config_list.append(config_diction)
        elif detect_mode == "0":
            config_diction={'col':col_name,'text_num':text_num,'recorded_data_list':recorded_data_list,'text_length':text_len_list,'text_type':text_type_list,'col_name_idx':idx,'col_data_idx':recorded_data_idx_list,'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data,'detect_mode':mode_check,'col_name_x':col_name_x,'col_name_y':col_name_y,'pass_num':pass_num}
            config=config_diction
            config_list.append(config_diction)    

    return recorded_data_list,recorded_data_idx_list
            
def first_compare(imformation_list,config_path,image_path)-> Union[List[dict],List[dict]]:
    img=cv2.imread(image_path)
    record_list=['QTY','LOT','DATE','PN','COO'] #紀錄事項會從config載入
    record_dict={'QTY':['OTY',"Q", "<QTY>", "Box Qty", "QTY", "QTY’", "QUANTITY", "Qty.", "Q’ ty", "Q’ty", "TOTALQTY", "Total Qty", "Unit Q’ty"],'LOT':["Bulk ID","1T", "<LOT>", "L/C", "L/N", "LN", "LOT", "LOT NO", "LOT NUMBER", "LOT NUMBERS", "LOT#", "LOTPO", "Lot Code", "Lot ID", "LOTNO", "Lot No.", "MLOT"],'DATE':["D","Trace code1","Trace codes","9D", "Assembly Date Code", "D/C", "DATE", "DATE CODE", "DATECODE", "DC", "DCODE", "DTE", "Seal Date"],'PN':["Type","1P", "P", "<P/N>", "CPN", "MPN", "P/N", "P/N Name", "PART", "PART ID", "PART NO", "PART NUMBER", "PN", "PROD ID", "PROD NO", "PRODUCT ID", "Part No.", "PartNo", "SPN"],'COO':["4L","Assembled In", "Assembly Location", "C.O.O.", "COO", "COUNTRY OF ORIGIN", "MADE IN", "Origin of"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    for col_list_name in record_dict.keys():
        col_list = record_dict[col_list_name]
        record_dict[col_list_name] = [col_name.upper() for col_name in col_list]
    result_dict={'QTY':[],'LOT':[],'DATE':[],'PN':[],'COO':[]} #紀錄比對事項
    result_list=[]
    match_text_list=[]
    config_list=[]
    label_num = 0
    pass_num_list=[]
    pass_num = 0
    for i,imformation in enumerate(imformation_list):
        match_text_flag=False
        text=imformation['text'] #找出目前的文字
        if text =="TRACE CODE1":
            print("hi")
        max_score_list=[]
        for col_list_name in record_dict.keys():#使用預設col_list_name查找圖片中相對應的col
            max_score=0
            offset=0
            col_list=record_dict[col_list_name]
            col_list=[col_name.upper() for col_name in col_list]
            #需再新增紀錄本類型圖片所使用的col名稱 降低失誤率
            #加入match_text機制
            for col_name in col_list:
                match_text=""
                score = fuzz.ratio(col_name,text)
                longest_match=difflib.SequenceMatcher(None, col_name, text).find_longest_match(0,len(col_name),0,len(text))
                if longest_match.size!=0:
                    match_text=col_name[longest_match.a:longest_match.a+longest_match.size]
                if score>max_score:
                    max_score=score
                    if max_score==100:
                        max_score_diction={'max_score':max_score,'col':col_list_name}
                        max_score_list.append(max_score_diction)
                        break                  
            if max_score>80 and match_text_flag==False: #如果分數達標 認定為col
                max_score_diction={'max_score':max_score,'col':col_list_name}#紀錄max_score分數
                max_score_list.append(max_score_diction)
        if len(max_score_list)!=0:
            max_score_list=sorted(max_score_list,key=lambda d:d['max_score'])
            col_list_name=max_score_list[-1]['col']
            result_dict[col_list_name].append(i)#紀錄在information list的位置

    saved_result_dict = copy.deepcopy(result_dict)
    for col_name in result_dict.keys():
        pass_num = 0
        #讀取config
        config=None
        #未來如果有預載入config可用
        for saved_config in config_list:
            if saved_config['col']==col_name:
                config=saved_config
        for idx in result_dict[col_name]:
            idx_imformation=imformation_list[idx]
            idx_col_data=idx_imformation['text']
            idx_col_x=idx_imformation['x']
            idx_col_y=idx_imformation['y']
            bounding_poly=idx_imformation['bounding_poly']
            jump_flag = False
            for diction in result_list:
                #偵測到為同一個bounding_poly 可能有問題
                if diction['bounding_poly'] == bounding_poly:
                    #偵測到是否存在此項目
                    keys = diction.keys()
                    for key in keys:
                        if col_name == key:
                            #認定存在
                            jump_flag = True
                            break
                    if jump_flag :
                        break
                if jump_flag :
                    break
            if jump_flag :
                continue
            min_x=None
            min_y=None
            max_x=None
            max_y=None

            for axis in bounding_poly:
                cur_x=axis[0]
                cur_y=axis[1]
                if min_x is None or min_x>cur_x:
                    min_x=cur_x
                if max_x is None or max_x<cur_x:
                    max_x=cur_x
                if min_y is None or min_y>cur_y:
                    min_y=cur_y
                if max_y is None or max_y<cur_y:
                    max_y=cur_y
            idx_img=img.copy()
            cv2.rectangle(idx_img, (min_x, min_y), (max_x, max_y), (0,0,255),thickness=3)
            idx_img=img_resize(idx_img)
            cv2.namedWindow("show_col", cv2.WINDOW_NORMAL)
            crop_y = int(1*idx_img.shape[0])
            crop_x = int(1*idx_img.shape[1])
            cv2.resizeWindow("show_col",crop_x,crop_y)
            cv2.imshow("show_col",idx_img)
            cv2.waitKey(1)
            print("請問是否偵測到"+str(col_name)+"之欄位? (在框框內即代表成功)")
            #print("欄位x座標:"+str(idx_col_x))
            #print("欄位y座標:"+str(idx_col_y))
            print("如果符合請輸入:T")
            check=input()
            if check.upper()!="T":
                pass_num = pass_num+1
                continue

            col_data_idx_list=search_col_data_idx(idx,20,imformation_list) #中間為查找data之範圍s
            

            #測試審核機制
            #config應針對不同選項有所調整 另外一個程式
            col_data,recorded_col_data_idx_list=compare_col_data(imformation_list,col_name,config_list,config,col_data_idx_list,idx,idx_col_data,idx_col_x,idx_col_y,pass_num)
            if col_data == None:
                continue
            # if len(col_data)<label_num :
            #     if len(col_data) == 1:
            #         print("目前共有"+len)
            #         print("請問是否所有"+col_name+"都是")
            for i in range(len(col_data)):
                if label_num<i:
                    label_num = i
                now_col_data = col_data[i]
                now_col_data_idx = recorded_col_data_idx_list[i]
                print("欄位資料:"+col_name+":"+now_col_data)
                print("欄位idx:"+str(now_col_data_idx+idx))
                diction={'col_name':col_name,col_name:now_col_data,"location":now_col_data_idx,"bounding_poly":bounding_poly,'label_id':0,'col_id':i}
                result_list.append(diction)



    result_list = check_result_list(result_list=result_list)
    result_list = sorted(result_list,key= lambda d:d['col_id'])
    result_list = sorted(result_list,key= lambda d:d['label_id'])

    #確認是否有遺漏
    for col_list_name in record_list:
        exist_flag = False
        for config in config_list:
            if config['col'] == col_list_name:
                exist_flag = True
                break
        if exist_flag == False:
            print("請問是否需要加入"+str(col_list_name)+"? 如果需要請輸入:T")
            check=input()
            if check.upper() =="T":
                print("開始找") #城市
                print("請輸入"+str(col_list_name)+"長度")
                text_length=int(input()) #給定長度搜尋
                for imformation in imformation_list:
                    now_text =imformation['text']
                    now_text_length = len(now_text)
                    if text_length == now_text_length:
                        print("請問是否"+str(now_text)+"為"+str(col_list_name)+"的data? 如果是請輸入:T")
                        confirm = input()
                        if confirm.upper() == "T":
                            config_diction={'col':col_list_name,'text_length':text_length,'text_type':'str','col_name_idx':-1,'col_data_idx':col_data_idx_list[i],'col_data_idx_list':col_data_idx_list,'col_name_text':'','col_data_text':now_text}
                            with open(os.path.join(config_path, "no_col_config.json"), "w") as f:
                                json.dump({"config": config_diction}, f)

    with open(os.path.join(config_path, "config.json"), "w") as f:
        json.dump({"config": config_list}, f)
    #加入第二種審核模式(提高準確率) 想法:word_result

    return result_list,match_text_list

def normal_compare(imformation_list,config,config_2,image_path)-> Union[List[dict],List[dict]]:

    img=cv2.imread(image_path)
    record_list=['QTY','LOT','DATE','PN','COO'] #紀錄事項會從config載入
    record_dict={'QTY':['OTY',"Q", "<QTY>", "Box Qty", "QTY", "QTY’", "QUANTITY", "Qty.", "Q’ ty", "Q’ty", "TOTALQTY", "Total Qty", "Unit Q’ty"],'LOT':["Bulk ID","1T", "<LOT>", "L/C", "L/N", "LN", "LOT", "LOT NO", "LOT NUMBER", "LOT NUMBERS", "LOT#", "LOTPO", "Lot Code", "Lot ID", "LOTNO", "Lot No.", "MLOT"],'DATE':["D","Trace code1","Trace codes","9D", "Assembly Date Code", "D/C", "DATE", "DATE CODE", "DATECODE", "DC", "DCODE", "DTE", "Seal Date"],'PN':["Type","1P", "P", "<P/N>", "CPN", "MPN", "P/N", "P/N Name", "PART", "PART ID", "PART NO", "PART NUMBER", "PN", "PROD ID", "PROD NO", "PRODUCT ID", "Part No.", "PartNo", "SPN"],'COO':["4L","Assembled In", "Assembly Location", "C.O.O.", "COO", "COUNTRY OF ORIGIN", "MADE IN", "Origin of"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    for col_list_name in record_dict.keys():
        col_list = record_dict[col_list_name]
        record_dict[col_list_name] = [col_name.upper() for col_name in col_list]
    result_dict={'QTY':[],'LOT':[],'DATE':[],'PN':[],'COO':[]} #紀錄比對事項
    result_list=[]
    match_text_list=[]
    config_list=config
    #確認目前col_name是否正確
    def check_col_name(now_col_name,now_idx,record_dict,record_col_name,imformation_list,pass_num) -> bool:
        #確認 now_idx 所對應的information是否為對應的 col_name

        max_num=50 #設定查找範圍
        back_num = 0
        forward_num = -1
        num = 0

        if len(imformation_list) <= now_idx:
            return False,-1
        match_col_name_list=[]
        while(forward_num<max_num or back_num<max_num):
            #往後找
            forward_num = forward_num+1
            if (len(imformation_list)-1)>=(now_idx+forward_num):
                now_imformation_text =imformation_list[now_idx+forward_num]['text']
                score = fuzz.ratio(record_col_name,now_imformation_text)
                if score>80:
                    return True,now_idx+forward_num
                    pass_num = pass_num -1
                # for col_name in record_dict[now_col_name]:
                # #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                #     score = fuzz.ratio(col_name,now_imformation_text)
                #     if now_imformation_text == col_name:
                #         return True,now_idx+forward_num
            #往前找
            back_num = back_num+1
            if (now_idx-back_num)>=0:
                now_imformation_text =imformation_list[now_idx-back_num]['text']
                score = fuzz.ratio(record_col_name,now_imformation_text)
                if score>80:
                    return True,now_idx-back_num
                    pass_num = pass_num-1
                for col_name in record_dict[now_col_name]:
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                    if now_imformation_text == col_name:
                        return True,now_idx-back_num
                    
        max_num=50 #設定查找範圍
        back_num = -1
        forward_num = -1
        num = 0

        if len(imformation_list) <= now_idx:
            return False,-1
        match_col_name_list=[]
        while(forward_num<max_num or back_num<max_num):
            #往後找
            forward_num = forward_num+1
            if (len(imformation_list)-1)>=(now_idx+forward_num):
                now_imformation_text =imformation_list[now_idx+forward_num]['text']
                # score = fuzz.ratio(record_col_name,now_imformation_text)
                # if score>80:
                #     return True,now_idx+forward_num
                for col_name in record_dict[now_col_name]:
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                    score = fuzz.ratio(col_name,now_imformation_text)
                    if now_imformation_text == col_name:
                        return True,now_idx+forward_num
            #往前找
            back_num = back_num+1
            if (now_idx-back_num)>=0:
                now_imformation_text =imformation_list[now_idx-back_num]['text']
                # score = fuzz.ratio(record_col_name,now_imformation_text)
                # if score>80:
                #     return True,now_idx-back_num
                for col_name in record_dict[now_col_name]:
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                    if now_imformation_text == col_name:
                        return True,now_idx-back_num
        return False,-1

    config_list=sorted(config_list,key=lambda d:d['col_name_idx'])
    x_imformation_list=imformation_list.copy()
    x_imformation_list =sorted(x_imformation_list,key=lambda d:d['x'])
    y_imformation_list=imformation_list.copy()
    label_id=-1
    col_id=-1
    x_config_list=[]
    y_config_list=[]

    for saved_config in config_list:
        if saved_config['detect_mode'] == "1":
            x_config_list.append(saved_config)
        else :
            y_config_list.append(saved_config)
    x_config_list = sorted(x_config_list,key= lambda d:d['col_name_x'])
    y_config_list = sorted(y_config_list,key= lambda d:d['col_name_y'])
    while len(y_imformation_list) != 0 and len(y_config_list)!=0:
        max_idx=-1
        label_id=label_id+1
        for saved_config in y_config_list:

            now_col_name = saved_config['col']
            now_idx = saved_config['col_name_idx']
            now_data_idx_list=saved_config['col_data_idx_list']
            record_col_name = saved_config['col_name_text']
            record_col_name_x = saved_config ['col_name_x']
            record_col_name_y = saved_config ['col_name_y']
            pass_num = saved_config['pass_num']
            col_name_flag = False
            if now_idx<(len(y_imformation_list)-1):
                if y_imformation_list[now_idx]['col_name_flag']:
                    for col_list_name in record_dict.keys():
                        col_list=record_dict[col_list_name]
                        col_list = [col_name.upper() for col_name in col_list]
                        for col_name in col_list:
                            if col_name == y_imformation_list[now_idx]['text']:
                                if now_col_name == col_list_name:
                                    col_name_flag = True
            if col_name_flag == False:
                col_name_flag,now_idx = check_col_name(now_col_name,now_idx,record_dict,record_col_name,y_imformation_list,pass_num)
            
            if col_name_flag == False:# 如為True 代表 一模一樣
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                continue

            if now_idx>max_idx:
                max_idx=now_idx

            pre_saved_config = copy.deepcopy(saved_config)
            recorded_col_data_list,recorded_col_data_idx_list = compare_col_data(y_imformation_list,now_col_name,config_list,pre_saved_config,now_data_idx_list,now_idx,record_col_name,record_col_name_x,record_col_name_y,pass_num)
            if recorded_col_data_list==None:
                continue
            for i in range(len(recorded_col_data_list)):
                col_data = recorded_col_data_list[i]
                col_data_idx = recorded_col_data_idx_list[i]
                #now_idx+i為col_data now_idx為col_name
                if col_data != None:
                    data_idx = now_idx+col_data_idx
                    if data_idx>max_idx:
                        max_idx=data_idx
                    col_name_imformation = imformation_list[now_idx]
                    col_y = col_name_imformation['y']
                    bounding_poly=col_name_imformation['bounding_poly']            
                    print("欄位資料:"+str(now_col_name)+":"+str(col_data))
                    #print("欄位y位置:"+str(col_y))
                    diction={'col_name':now_col_name,now_col_name:col_data,"location":col_y,"bounding_poly":bounding_poly,"label_id":label_id,'col_id':i}
                    result_list.append(diction)

        y_imformation_list=y_imformation_list[max_idx+1:]
        if max_idx == -1:
            break
    label_id = -1
    while len(x_imformation_list) != 0 and len(x_config_list)!=0:
        max_idx=-1
        label_id=label_id+1

        for saved_config in x_config_list:
            col_name_flag = False
            now_col_name = saved_config['col']
            now_idx = saved_config['col_name_idx']

            now_data_idx_list=saved_config['col_data_idx_list']
            record_col_name = saved_config['col_name_text']
            record_col_name_x = saved_config ['col_name_x']
            record_col_name_y = saved_config ['col_name_y']
            pass_num = saved_config['pass_num']
            if now_idx<(len(x_imformation_list)-1):
                if x_imformation_list[now_idx]['col_name_flag']:
                    for col_list_name in record_dict.keys():
                        col_list=record_dict[col_list_name]
                        col_list = [col_name.upper() for col_name in col_list]
                        for col_name in col_list:
                            if col_name == x_imformation_list[now_idx]['text']:
                                col_name_flag = True
            if col_name_flag == False:
                col_name_flag,now_idx = check_col_name(now_col_name,now_idx,record_dict,record_col_name,x_imformation_list,pass_num)
            
            if col_name_flag == False:# 如為True 代表 一模一樣
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                continue

            if now_idx>max_idx:
                max_idx=now_idx

            pre_saved_config = copy.deepcopy(saved_config)
            recorded_col_data_list,recorded_col_data_idx_list = compare_col_data(x_imformation_list,now_col_name,config_list,pre_saved_config,now_data_idx_list,now_idx,record_col_name,record_col_name_x,record_col_name_y,pass_num)
            if recorded_col_data_list==None:
                continue
            for i in range(len(recorded_col_data_list)):
                col_data = recorded_col_data_list[i]
                col_data_idx = recorded_col_data_idx_list[i]
                #now_idx+i為col_data now_idx為col_name
                if col_data != None:
                    data_idx = now_idx+col_data_idx
                    if data_idx>max_idx:
                        max_idx=data_idx
                    col_name_imformation = imformation_list[now_idx]
                    col_y = col_name_imformation['y']
                    bounding_poly=col_name_imformation['bounding_poly']            
                    print("欄位資料:"+str(now_col_name)+":"+str(col_data))
                    #print("欄位y位置:"+str(col_y))
                    diction={'col_name':now_col_name,now_col_name:col_data,"location":col_y,"bounding_poly":bounding_poly,"label_id":label_id,'col_id':i}
                    result_list.append(diction)

        x_imformation_list=x_imformation_list[max_idx+1:]
        if max_idx == -1:
            break

    #加入第二種審核模式用於標點符號沒偵測到
    result_list = check_result_list(result_list=result_list)
    result_list = sorted(result_list,key= lambda d:d['col_id'])
    result_list = sorted(result_list,key= lambda d:d['label_id'])
    top_ans_text_list=[]
    if config_2!=None:
        config_text = config_2['col_data_text']
        config_text_length = config_2['text_length']
        possible_ans=[]
        ignore_flag = False
        for imformation in imformation_list:
            ignore_flag = False
            imformation_text = imformation['text']
            imformation_text_y = imformation['y']
            imformation_text_length = len(imformation_text)
            if imformation_text_length == config_text_length:
                for result in result_list:
                    col_name = result['col_name']
                    col_text = result[col_name]
                    if col_text == imformation_text:
                        ignore_flag = True
                        break
                if ignore_flag == False:
                    diction={'text':imformation_text,'y':imformation_text_y}
                    possible_ans.append(diction)
        max_score = 0
        ans_text_list=[]
        
        max_label_id = int(result_list[-1]['label_id'])+1
        for text_diction in possible_ans:
            text = text_diction['text']
            text_y = text_diction['y']
            score = fuzz.ratio(text,config_text)
            diction={'text':text,'score':score,'y':text_y}
            ans_text_list.append(diction)
        ans_text_list=sorted(ans_text_list,key=lambda d:d['score'])
        top_ans_text_list=[]
        for i in range(max_label_id):
            ans_text = ans_text_list[(len(ans_text_list)-1-i)]
            top_ans_text_list.append(ans_text)
        top_ans_text_list=sorted(top_ans_text_list,key=lambda d:d['y'])
    return result_list,top_ans_text_list

# barcode找不到未設計完成
def barcode_compare_ocr(result_list,dbr_decode_res):#要改

    record_list=['QTY','LOT','DATE','PN','COO']
    barcode_list = dbr_decode_res
    del_barcode_idx=-1
    now_label_id=0
    saved_barcode_list=barcode_list.copy()
    combined_result=[]
    match_text_list=[]
    for result in result_list:
        #換下一張label compare to barcode result
        del_barcode_idx=-1
        match_text_list=[]
        closet_diction = None
        highest_score_diction = None
        if result['label_id'] != now_label_id:
            combined_result=sorted(combined_result,key=lambda d:d['label_id'])
            for combined_res in combined_result:
                if combined_res['del_barcode_idx']==-1:
                    continue
                if combined_res['del_barcode_idx']>del_barcode_idx and combined_res['label_id'] == now_label_id:
                    del_barcode_idx = combined_res['del_barcode_idx']
            now_label_id=result['label_id']
            barcode_list = barcode_list[del_barcode_idx+1:]

        for col_name in record_list:
            if col_name in result:
                both_exist_flag=False
                ocr_bounding_poly = result['bounding_poly']
                now_col_id = result['col_id']
                col_data = result[col_name]
                ocr_y = result['location']
                for idx,barcode_result in enumerate(barcode_list):
                    barcode_text = barcode_result['text']
                    barcode_bounding_poly = barcode_result['bounding_poly']
                    bar_y = int((barcode_bounding_poly[1][1]+barcode_bounding_poly[2][1])/2)
                    score = fuzz.ratio(col_data,barcode_text) #評分機制很差 要再改
                    longest_match=difflib.SequenceMatcher(None, col_data, barcode_text).find_longest_match(0,len(col_data),0,len(barcode_text))
                    match_text = None
                    if longest_match.size!=0:
                        match_text=col_data[longest_match.a:longest_match.a+longest_match.size]
                    if score>60:
                        if del_barcode_idx<idx:
                            del_barcode_idx = idx
                        diction={'col_name':col_name,'ocr_result':col_data,'barcode_result':barcode_text,'label_id':now_label_id,'col_id':now_col_id,'color_idx':-1,'barcode_bounding_poly':barcode_bounding_poly,'ocr_bounding_poly':ocr_bounding_poly,'del_barcode_idx':idx,'ocr_y':ocr_y,'bar_y':bar_y}
                        match_diction={"del_barcode_idx":del_barcode_idx,'result_information':diction,'match_score':score}
                        match_text_list.append(match_diction)
                        #combined_result.append(diction)
                if len(match_text_list) != 0:
                    both_exist_flag = True
                    max_score = 0
                    match_text_list=sorted(match_text_list,key=lambda d:d['match_score'])
                    min_diff_y = 1000000
                    for diction in match_text_list:
                        result_imformation = diction['result_information']
                        now_col_id = result_imformation['col_id']
                        ocr_y= result_imformation['ocr_y']
                        bar_y = result_imformation['bar_y']
                        now_diff_y = abs(ocr_y-bar_y)
                        #分數夠高用 score 仍只靠上下順序關係 一有miss情形就錯誤
                        if diction['match_score']>max_score and diction['match_score']>60:
                            highest_score_diction = diction['result_information']
                            max_score = diction['match_score']
                            del_barcode_idx = diction['del_barcode_idx']
                        else:
                            if now_diff_y<min_diff_y: #min_diff_y 要做調整用於在有條碼沒辨識到時能夠停止
                                min_diff_y = now_diff_y
                                closet_diction = diction['result_information']

                    if highest_score_diction == None and closet_diction != None:
                        highest_score_diction = closet_diction
                        
    
                    combined_result.append(highest_score_diction)
                if both_exist_flag==False:
                    diction={'col_name':col_name,'col_id':now_col_id,'label_id':now_label_id,"color_idx":-1,'ocr_result':col_data,'barcode_result':"no barcode result",'barcode_bounding_poly':ocr_bounding_poly,'ocr_bounding_poly':ocr_bounding_poly,'del_barcode_idx':-1 ,'ocr_y':ocr_y,'bar_y':-1}
                    combined_result.append(diction)
    return combined_result
            


        

    
