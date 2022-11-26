from typing import List,Union
import difflib
import numpy as np
import json
import os
import copy
import cv2
from fuzzywuzzy import fuzz

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
    #處理data
    for j in range(len(results)):
        location=results[j][0] #文字方框位置
        location_mid_x=int((location[0][0]+location[1][0])/2) #找出方框x中點
        location_mid_y=int((location[1][1]+location[2][1])/2) #找出方框y中點
        text=results[j][1][0].upper()#找出偵測文字
        #text=text.replace(" ","")
        score=results[j][1][1]#找出文字分數

        #標點符號分字
        split_flag=False
        if text.find(' ') != -1:#可能要再改
            split_flag=True
            text_list=text.split(" ")
            for text in text_list:
                cur_text=text
                if cur_text!='':
                    diction={'x':location_mid_x,'y':location_mid_y,'text':cur_text,'bounding_poly':location,'score':score}
                    imformation_list.append(diction)
        if split_flag==False:    
            diction={'x':location_mid_x,'y':location_mid_y,'text':text,'bounding_poly':location,'score':score}
            imformation_list.append(diction)
        # if diction['text']!='':
        #     imformation_list.append(diction)
    imformation_list=sorted(imformation_list,key=lambda d:d['y'])#由y軸座標排序

    return imformation_list

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
def compare_col_data(col_data_list,col_name,config_list,config,col_data_idx_list,idx,idx_col_data):
    col_data_list = sorted(col_data_list,key=lambda d:d['diff_y'])
    for i,col_dict in enumerate(col_data_list):
        col_data = col_dict['col_data']
        data=col_name+":"+col_data
        if config==None:
            print(data)
            print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
            check=input()
            if check=="T":
                text_len=len(col_data) #儲存長度參數
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type='int'
                print(text_type)
                config_diction={'col':col_name,'text_length':text_len,'text_type':text_type,'col_name_idx':idx,'col_data_idx':col_data_idx_list[i],'col_data_idx_list':col_data_idx_list,'col_name_text':idx_col_data}
                config=config_diction
                config_list.append(config_diction)
                break
        else:
            #有config
            text_len=len(col_data)
            text_type=type(col_data).__name__
            if text_type=='str':
                if col_data.isdigit():
                    text_type='int'
            if text_type == config['text_type']:

                if text_type=='str':
                    if text_len != config['text_length']:
                        continue
                    else:
                        break
            else:
                continue
            break

    return col_data,i

###要將col_name一樣屬性的包一起###
def first_compare(imformation_list,config_path,image_path)-> Union[List[dict],List[dict]]:
    img=cv2.imread(image_path)
    record_list=['QTY','LOT','DATE','PN','COO'] #紀錄事項會從config載入
    record_dict={'QTY':["Q", "<QTY>", "Box Qty", "QTY", "QTY’", "QUANTITY", "Qty", "Qty.", "Quantity", "Q’ ty", "Q’TY", "Q’ty", "TOTALQTY", "Total Qty", "Unit Q’ty"],'LOT':["1T", "<LOT>", "L/C", "L/N", "LN", "LOT", "LOT ID", "LOT NO", "LOT NUMBER", "LOT NUMBERS", "LOT#", "LOTPO", "Lot", "Lot Code", "Lot ID", "LOTNO", "Lot No", "Lot No.", "Lot Number", "LotNo", "MLOT"],'DATE':["9D", "D", "Assembly Date Code", "D/C", "DATE", "DATE CODE", "DATECODE", "DC", "DCODE", "DTE", "Date", "Date Code", "Date code", "DateCode", "Seal Date"],'PN':["1P", "P", "<P/N>", "CPN", "MPN", "P/N", "P/N Name", "PART", "PART ID", "PART NO", "PART NUMBER", "PN", "PROD ID", "PROD NO", "PRODUCT ID", "Part", "Part No", "Part No.", "Part Number", "PartNo", "Product Id", "Product id", "SPN"],'COO':["4L", "COO", "Assembled In", "Assembly Location", "C.O.O.", "COO", "COUNTRY OF ORIGIN", "CoO", "Coo", "Country of Origin", "Country of origin", "MADE IN", "Made in", "Origin of"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    result_dict={'QTY':[],'LOT':[],'DATE':[],'PN':[],'COO':[]} #紀錄比對事項
    result_list=[]
    match_text_list=[]
    config_list=[]

    for i,imformation in enumerate(imformation_list):
        match_text_flag=False
        text=imformation['text'] #找出目前的文字
        max_score_list=[]
        for col_list_name in record_dict.keys():#使用預設col_list_name查找圖片中相對應的col
            max_score=0
            offset=0
            col_list=record_dict[col_list_name]
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
                        break
                #match_text機制
                #如match_text後有文字才新增
                    
            #     if match_text==col_name and match_text!=text[longest_match.b:]:
            #         match_text_flag=True
            #         #print(text[longest_match.b+longest_match.size:])
            #         offset=offset+1
            #         new_text=text[(longest_match.b+longest_match.size):]
            #         #判斷是否有因":"分詞，有的話代表有贅詞
            #         if i!=0:
            #             if(imformation_list[i-1]['x']==imformation_list[i]['x'] and imformation_list[i-1]['y']==imformation_list[i]['y']):
            #                 new_text=imformation_list[i-1]['text']
            #         new_information_dict={'text':new_text,'idx':i,'col':col_list_name}
            #         match_text_list.append(new_information_dict)
            #         break
            #     #match_text無新增 但文字符合col_name 高度正相關
            #     if match_text==col_name:
            #         max_score=2
            # if match_text_flag:
            #     break
                    
            if max_score>80 and match_text_flag==False: #如果分數達標 認定為col
                max_score_diction={'max_score':max_score,'col':col_list_name}#紀錄max_score分數
                max_score_list.append(max_score_diction)
        if len(max_score_list)!=0:
            max_score_list=sorted(max_score_list,key=lambda d:d['max_score'])
            col_list_name=max_score_list[-1]['col']
            result_dict[col_list_name].append(i)#紀錄在information list的位置

    saved_result_dict = copy.deepcopy(result_dict)
    for col_name in result_dict.keys():
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
            if check!="T":
                continue

            col_data_idx_list=search_col_data_idx(idx,20,imformation_list) #中間為查找data之範圍
            col_data_list=[]
            #col_data查找範圍
            for data_idx in col_data_idx_list:
                if data_idx+idx<len(imformation_list):
                    col_data = imformation_list[idx+data_idx]['text']
                    diff_y = imformation_list[idx]['y']-imformation_list[idx+data_idx]['y']
                    diction={'col_data':col_data,'diff_y':abs(diff_y),'idx':idx+data_idx}
                    col_data_list.append(diction)
                else:
                    break
            

            #測試審核機制
            #config應針對不同選項有所調整 另外一個程式
            col_data,i=compare_col_data(col_data_list,col_name,config_list,config,col_data_idx_list,idx,idx_col_data)

            print("欄位資料:"+col_name+":"+col_data)
            print("欄位idx:"+str(col_data_idx_list[i]+idx))
            diction={col_name:col_data,"location":col_data_idx_list[i],"bounding_poly":bounding_poly,'label_id':0}
            result_list.append(diction)

    with open(os.path.join(config_path, "config.json"), "w") as f:
        json.dump({"config": config_list}, f)
    #加入第二種審核模式(提高準確率) 想法:word_result

    return result_list,match_text_list

def normal_compare(imformation_list,config,image_path)-> Union[List[dict],List[dict]]:

    img=cv2.imread(image_path)
    record_list=['QTY','lot','Date'] #紀錄事項會從config載入
    record_dict={'QTY':["Q", "<QTY>", "Box Qty", "QTY", "QTY’", "QUANTITY", "Qty", "Qty.", "Quantity", "Q’ ty", "Q’TY", "Q’ty", "TOTALQTY", "Total Qty", "Unit Q’ty"],'LOT':["1T", "<LOT>", "L/C", "L/N", "LN", "LOT", "LOT ID", "LOT NO", "LOT NUMBER", "LOT NUMBERS", "LOT#", "LOTPO", "Lot", "Lot Code", "Lot ID", "LOTNO", "Lot No", "Lot No.", "Lot Number", "LotNo", "MLOT"],'DATE':["9D", "D", "Assembly Date Code", "D/C", "DATE", "DATE CODE", "DATECODE", "DC", "DCODE", "DTE", "Date", "Date Code", "Date code", "DateCode", "Seal Date"],'PN':["1P", "P", "<P/N>", "CPN", "MPN", "P/N", "P/N Name", "PART", "PART ID", "PART NO", "PART NUMBER", "PN", "PROD ID", "PROD NO", "PRODUCT ID", "Part", "Part No", "Part No.", "Part Number", "PartNo", "Product Id", "Product id", "SPN"],'COO':["4L", "COO", "Assembled In", "Assembly Location", "C.O.O.", "COO", "COUNTRY OF ORIGIN", "CoO", "Coo", "Country of Origin", "Country of origin", "MADE IN", "Made in", "Origin of"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    result_dict={'QTY':[],'LOT':[],'DATE':[],'PN':[],'COO':[]} #紀錄比對事項
    result_list=[]
    match_text_list=[]
    config_list=config

    #確認目前col_name是否正確
    def check_col_name(now_col_name,now_idx,record_dict,record_col_name,imformation_list) -> bool:
        #確認 now_idx 所對應的information是否為對應的 col_name

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
                score = fuzz.ratio(record_col_name,now_imformation_text)
                if score>80:
                    return True,now_idx+forward_num
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
                # for col_name in record_dict[now_col_name]:
                # #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                #     if now_imformation_text == col_name:
                #         return True,now_idx-back_num
                    
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
    saved_imformation_list=imformation_list.copy()
    label_id=-1
    while len(imformation_list) != 0:
        max_idx=-1
        label_id=label_id+1

        for saved_config in config_list:

            now_col_name = saved_config['col']
            now_idx = saved_config['col_name_idx']
            now_data_idx=saved_config['col_data_idx']
            now_data_idx_list=saved_config['col_data_idx_list']
            record_col_name = saved_config['col_name_text']
            col_name_flag,now_idx = check_col_name(now_col_name,now_idx,record_dict,record_col_name,imformation_list)
            
            if col_name_flag == False:# 如為True 代表 一模一樣
                #加入評分機制for 確認 now_col_name 與 imformation_list中 text 的符合程度
                continue

            if now_idx>max_idx:
                max_idx=now_idx

            col_data_list=[]

            for data_idx in now_data_idx_list:
                # 將config中data_idx_list對應到的data取出
                if data_idx+now_idx<len(imformation_list):
                    col_data = imformation_list[now_idx+data_idx]['text']
                    diff_y = imformation_list[now_idx]['y']-imformation_list[now_idx+data_idx]['y']
                    diction={'col_data':col_data,'diff_y':abs(diff_y)}
                    col_data_list.append(diction)
                else:
                    break

            col_data,i = compare_col_data(col_data_list,now_col_name,config_list,saved_config,now_data_idx_list,now_idx,record_col_name)
            #now_idx+i為col_data now_idx為col_name
            col_name_imformation = imformation_list[now_idx]
            col_y = col_name_imformation['y']
            bounding_poly=col_name_imformation['bounding_poly']            
            print("欄位資料:"+str(now_col_name)+":"+str(col_data))
            #print("欄位y位置:"+str(col_y))
            diction={now_col_name:col_data,"location":col_y,"bounding_poly":bounding_poly,"label_id":label_id}
            result_list.append(diction)

        imformation_list=imformation_list[max_idx+1:]
        if max_idx == -1:
            break

    #result_grouping(result_list,img)
        
    #def word_grouping():
    #match的不使用config限制 暫定
    #可能加入判斷後 再確認鄰近information判別是否有符合之資訊
    # for match_diction in match_text_list:
    #     config=None
    #     col_name=match_diction['col']
    #     for saved_config in config_list:
    #         if saved_config['col']==col_name:
    #             config=saved_config
    #     #載入資訊
    #     idx=match_diction['idx']
    #     col_information=imformation_list[idx]
    #     col_y=col_information['y']
    #     col_data=match_diction['text']
    #     data=col_name+":"+col_data
    #     if config==None:
    #         print(data)
    #         print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
    #         check=input()
    #         if check=="T":
    #             text_len=len(col_data) #儲存長度參數
    #             #print(type(col_data))
    #             text_type=type(col_data).__name__
    #             if text_type=='str':
    #                 if col_data.isdigit():
    #                     text_type='int'
    #             config_diction={'col':col_name,'text_length':text_len,'text_type':text_type}
    #             config=config_diction
    #             config_list.append(config_diction)
    #         else:
    #             match_text_list.remove(match_diction)
    #             continue
    #     else:
    #         text_len=len(col_data)
    #         text_type=type(col_data).__name__
    #         if text_type=='str':
    #             if col_data.isdigit():
    #                 text_type='int'
    #         if text_type == config['text_type']:
    #             if text_type=='str':
    #                 if text_len != config['text_length']:
    #                     continue
    #         else:
    #             continue
    #     print("欄位資料:"+data)
    #     print("欄位y位置:"+str(col_y))
    #     diction={col_name:col_data,"location":col_y}
    #     result_list.append(diction)

    #加入第二種審核模式用於標點符號沒偵測到
    return result_list,match_text_list

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
                    if match_text == col_data:
                        if del_barcode_idx<idx:
                            del_barcode_idx = idx
                        diction={'col_name':col_name,'ocr_result':col_data,'barcode_result':barcode_text,'label_id':now_label_id,'barcode_bounding_poly':barcode_bounding_poly,'ocr_bounding_poly':ocr_bounding_poly,'del_barcode_idx':idx,'ocr_y':ocr_y,'bar_y':bar_y}
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
                    diction={'col_name':col_name,'ocr_result':col_name+":"+col_data,'barcode_result':"no barcode result",'label_id':now_label_id,'barcode_bounding_poly':ocr_bounding_poly,'ocr_bounding_poly':ocr_bounding_poly,'del_barcode_idx':-1 ,'ocr_y':ocr_y,'bar_y':-1}
                    combined_result.append(diction)
    return combined_result
            


        

    
