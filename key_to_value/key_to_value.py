from typing import List,Union
import difflib
import numpy as np
import json
import os

#觀察:(XX) XX為用於qrcode中的標示
def data_preprocess(results)-> List[dict]:#傳入圖片ocr辨識結果  
    imformation_list=[]
    #處理data
    for j in range(len(results)):
        location=results[j][0] #文字方框位置
        location_mid_x=int((location[0][0]+location[1][0])/2) #找出方框x中點
        location_mid_y=int((location[1][1]+location[2][1])/2) #找出方框y中點
        text=results[j][1][0].upper()#找出偵測文字
        text=text.replace(" ","")
        score=results[j][1][1]#找出文字分數
        if text.find(':') != -1:#可能要再改
            text_list=text.split(":")
            sec_text=text_list[1]
            text=text_list[0]
            if sec_text!='':
                diction={'x':location_mid_x,'y':location_mid_y,'text':sec_text,'score':score}
                imformation_list.append(diction)
        if text.find('.') != -1:#可能要再改
            text_list=text.split(".")
            sec_text=text_list[1]
            text=text_list[0]
            if sec_text!='':
                diction={'x':location_mid_x,'y':location_mid_y,'text':sec_text,'score':score}
                imformation_list.append(diction)
        
        diction={'x':location_mid_x,'y':location_mid_y,'text':text,'score':score}
        imformation_list.append(diction)
    imformation_list=sorted(imformation_list,key=lambda d:d['y'])#由y軸座標排序

    return imformation_list

def barcode_data_preprocess(codes)-> List[dict]:#傳入圖片ocr辨識結果  
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

def first_compare(imformation_list,config_path)-> Union[List[dict],List[dict]]:
    record_list=['QTY','lot'] #紀錄事項會從config載入
    record_dict={'QTY':["QUANTITY","TOTALQTY","QTY"],'LOT':["LOTNO","LOTID","LOT"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    result_dict={'QTY':[],'LOT':[]} #紀錄比對事項
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
                score=difflib.SequenceMatcher(None, col_name, text).quick_ratio()
                longest_match=difflib.SequenceMatcher(None, col_name, text).find_longest_match(0,len(col_name),0,len(text))
                if longest_match.size!=0:
                    match_text=col_name[longest_match.a:longest_match.a+longest_match.size]
                if score>max_score:
                    max_score=score
                    if max_score==1.0:
                        break
                #match_text機制
                #如match_text後有文字才新增
                if match_text==col_name and match_text!=text[longest_match.b:]:
                    match_text_flag=True
                    #print(text[longest_match.b+longest_match.size:])
                    offset=offset+1
                    new_text=text[(longest_match.b+longest_match.size):]
                    #判斷是否有因":"分詞，有的話代表有贅詞
                    if i!=0:
                        if(imformation_list[i-1]['x']==imformation_list[i]['x'] and imformation_list[i-1]['y']==imformation_list[i]['y']):
                            new_text=imformation_list[i-1]['text']
                    new_information_dict={'text':new_text,'idx':i,'col':col_list_name}
                    match_text_list.append(new_information_dict)
                    break
                #match_text無新增 但文字符合col_name 高度正相關
                if match_text==col_name:
                    max_score=2
            if match_text_flag:
                break
            if max_score>0.6 and match_text_flag==False: #如果分數達標 認定為col
                max_score_diction={'max_score':max_score,'col':col_list_name}#紀錄max_score分數
                max_score_list.append(max_score_diction)
        if len(max_score_list)!=0:
            max_score_list=sorted(max_score_list,key=lambda d:d['max_score'])
            col_list_name=max_score_list[-1]['col']
            result_dict[col_list_name].append(i)#紀錄在information list的位置


    for col_name in result_dict.keys():
        #讀取config
        config=None
        #未來如果有預載入config可用
        for saved_config in config_list:
            if saved_config['col']==col_name:
                config=saved_config
        for idx in result_dict[col_name]:
            pre_information=None
            next_information=None
            col_data=None
            col_information=imformation_list[idx]
            col_x=col_information['x']
            col_y=col_information['y']
            if idx != 0:
                pre_idx=idx-1
                while(imformation_list[pre_idx]['x']<imformation_list[idx]['x']) and pre_idx != 0:
                    pre_idx=pre_idx-1
                pre_information=imformation_list[pre_idx]
            if idx !=(len(imformation_list)-1):
                next_idx=idx+1
                while(imformation_list[next_idx]['x']<imformation_list[idx]['x']) and next_idx != (len(imformation_list)-1):
                    next_idx=next_idx+1
                next_information=imformation_list[next_idx]
            if pre_information!=None and next_information!=None:
                pre_x=pre_information['x']
                pre_y=pre_information['y']
                next_x=next_information['x']
                next_y=next_information['y']
                if (col_y-pre_y)<=(next_y-col_y):#比較y軸差距
                    diff_y=col_y-pre_y
                    col_data=pre_information['text']
                    #下方人工審核模式先跳過
                    # if(col_x==pre_x): # 來自同一次detect
                    #     col_data=pre_information['text']
                    # elif(pre_x-col_x)>diff_y*10:
                    #     print("進入人工審核")
                else:
                    col_data=next_information['text']
            elif pre_information!=None:
                col_data=pre_information['text']
            else:
                col_data=next_information['text']
            data=col_name+":"+col_data
            #測試審核機制
            #config應針對不同選項有所調整 另外一個程式
            if config==None:
                print(data)
                print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
                check=input()
                if check=="T":
                    text_len=len(col_data) #儲存長度參數
                    print(type(col_data))
                    text_type=type(col_data).__name__
                    if text_type=='str':
                        if col_data.isdigit():
                            text_type=='int'
                    config_diction={'col':col_name,'text_length':text_len,'text_type':text_type}
                    config=config_diction
                    config_list.append(config_diction)
                else:
                    result_dict[col_name].remove(idx)
                    continue
            else:
                #有config
                text_len=len(col_data)
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type=='int'
                if text_type == config['text_type']:
                    if text_type=='str':
                        if text_len != config['text_length']:
                            continue
            
            print("欄位資料:"+data)
            print("欄位y位置:"+str(col_y))
            diction={"data":data,"location":col_y}
            result_list.append(diction)
    
    #match的不使用config限制 暫定
    #可能加入判斷後 再確認鄰近information判別是否有符合之資訊
    for match_diction in match_text_list:
        config=None
        col_name=match_diction['col']
        for saved_config in config_list:
            if saved_config['col']==col_name:
                config=saved_config
        #載入資訊
        idx=match_diction['idx']
        col_information=imformation_list[idx]
        col_y=col_information['y']
        col_data=match_diction['text']
        data=col_name+":"+col_data
        if config==None:
            print(data)
            print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
            check=input()
            if check=="T":
                text_len=len(col_data) #儲存長度參數
                #print(type(col_data))
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type='int'
                config_diction={'col':col_name,'text_length':text_len,'text_type':text_type}
                config=config_diction
                config_list.append(config_diction)
            else:
                match_text_list.remove(match_diction)
                continue
        else:
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
                continue
        print("欄位資料:"+data)
        print("欄位y位置:"+str(col_y))
        diction={"data":data,"location":col_y}
        result_list.append(diction)
    
    with open(os.path.join(config_path, "config.json"), "w") as f:
        json.dump({"config": config_list}, f)
    #加入第二種審核模式用於標點符號沒偵測到
    return result_list,match_text_list

#def normal_compare(config):
def normal_compare(imformation_list,config)-> Union[List[dict],List[dict]]:
    record_list=['QTY','lot'] #紀錄事項會從config載入
    record_dict={'QTY':["QUANTITY","TOTALQTY","QTY"],'LOT':["LOTNO","LOTID","LOT"]} #使用record_list紀錄事項可能名稱 需由長到短排序 for match_text
    result_dict={'QTY':[],'LOT':[]} #紀錄比對事項
    result_list=[]
    match_text_list=[]
    config_list=config

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
                score=difflib.SequenceMatcher(None, col_name, text).quick_ratio()
                longest_match=difflib.SequenceMatcher(None, col_name, text).find_longest_match(0,len(col_name),0,len(text))
                if longest_match.size!=0:
                    match_text=col_name[longest_match.a:longest_match.a+longest_match.size]
                if score>max_score:
                    max_score=score
                    if max_score==1.0:
                        break
                #match_text機制
                #如match_text後有文字才新增
                if match_text==col_name and match_text!=text[longest_match.b:]:
                    match_text_flag=True
                    #print(text[longest_match.b+longest_match.size:])
                    offset=offset+1
                    new_text=text[(longest_match.b+longest_match.size):]
                    #判斷是否有因":"分詞，有的話代表有贅詞
                    if i!=0:
                        if(imformation_list[i-1]['x']==imformation_list[i]['x'] and imformation_list[i-1]['y']==imformation_list[i]['y']):
                            new_text=imformation_list[i-1]['text']
                    new_information_dict={'text':new_text,'idx':i,'col':col_list_name}
                    match_text_list.append(new_information_dict)
                    break
                #match_text無新增 但文字符合col_name 高度正相關
                if match_text==col_name:
                    max_score=2
            if match_text_flag:
                break
            if max_score>0.6 and match_text_flag==False: #如果分數達標 認定為col
                max_score_diction={'max_score':max_score,'col':col_list_name}#紀錄max_score分數
                max_score_list.append(max_score_diction)
        if len(max_score_list)!=0:
            max_score_list=sorted(max_score_list,key=lambda d:d['max_score'])
            col_list_name=max_score_list[-1]['col']
            result_dict[col_list_name].append(i)#紀錄在information list的位置


    for col_name in result_dict.keys():
        #讀取config
        config=None
        #未來如果有預載入config可用
        for saved_config in config_list:
            if saved_config['col']==col_name:
                config=saved_config
        for idx in result_dict[col_name]:
            pre_information=None
            next_information=None
            col_data=None
            col_information=imformation_list[idx]
            col_x=col_information['x']
            col_y=col_information['y']
            if idx != 0:
                pre_idx=idx-1
                while(imformation_list[pre_idx]['x']<imformation_list[idx]['x']) and pre_idx != 0:
                    pre_idx=pre_idx-1
                pre_information=imformation_list[pre_idx]
            if idx !=(len(imformation_list)-1):
                next_idx=idx+1
                while(imformation_list[next_idx]['x']<imformation_list[idx]['x']) and next_idx != (len(imformation_list)-1):
                    next_idx=next_idx+1
                next_information=imformation_list[next_idx]
            if pre_information!=None and next_information!=None:
                pre_x=pre_information['x']
                pre_y=pre_information['y']
                next_x=next_information['x']
                next_y=next_information['y']
                if (col_y-pre_y)<=(next_y-col_y):#比較y軸差距
                    diff_y=col_y-pre_y
                    col_data=pre_information['text']
                    #下方人工審核模式先跳過
                    # if(col_x==pre_x): # 來自同一次detect
                    #     col_data=pre_information['text']
                    # elif(pre_x-col_x)>diff_y*10:
                    #     print("進入人工審核")
                else:
                    col_data=next_information['text']
            elif pre_information!=None:
                col_data=pre_information['text']
            else:
                col_data=next_information['text']
            data=col_name+":"+col_data
            #測試審核機制
            #config應針對不同選項有所調整 另外一個程式
            if config==None:
                print(data)
                print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
                check=input()
                if check=="T":
                    text_len=len(col_data) #儲存長度參數
                    print(type(col_data))
                    text_type=type(col_data).__name__
                    if text_type=='str':
                        if col_data.isdigit():
                            text_type=='int'
                    config_diction={'col':col_name,'text_length':text_len,'text_type':text_type}
                    config=config_diction
                    config_list.append(config_diction)
                else:
                    result_dict[col_name].remove(idx)
                    continue
            else:
                #有config
                text_len=len(col_data)
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type=='int'
                if text_type == config['text_type']:
                    if text_type=='str':
                        if text_len != config['text_length']:
                            continue
            
            print("欄位資料:"+data)
            print("欄位y位置:"+str(col_y))
            diction={"data":data,"location":col_y}
            result_list.append(diction)
    
    #match的不使用config限制 暫定
    #可能加入判斷後 再確認鄰近information判別是否有符合之資訊
    for match_diction in match_text_list:
        config=None
        col_name=match_diction['col']
        for saved_config in config_list:
            if saved_config['col']==col_name:
                config=saved_config
        #載入資訊
        idx=match_diction['idx']
        col_information=imformation_list[idx]
        col_y=col_information['y']
        col_data=match_diction['text']
        data=col_name+":"+col_data
        if config==None:
            print(data)
            print("請問是否符合"+col_name+"格式?如果符合請輸入:T")
            check=input()
            if check=="T":
                text_len=len(col_data) #儲存長度參數
                #print(type(col_data))
                text_type=type(col_data).__name__
                if text_type=='str':
                    if col_data.isdigit():
                        text_type='int'
                config_diction={'col':col_name,'text_length':text_len,'text_type':text_type}
                config=config_diction
                config_list.append(config_diction)
            else:
                match_text_list.remove(match_diction)
                continue
        else:
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
                continue
        print("欄位資料:"+data)
        print("欄位y位置:"+str(col_y))
        diction={"data":data,"location":col_y}
        result_list.append(diction)

    #加入第二種審核模式用於標點符號沒偵測到
    return result_list,match_text_list
            
            


        

    
