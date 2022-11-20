from key_to_value import key_to_value
import json
import os

def ocr_to_result(para_ocr_result):    
    imformation_list=key_to_value.data_preprocess(para_ocr_result)
    config=None
    image_path="./result_dir/"
    config_path=image_path+"config.json"
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config=json.load(f)['config']
    if config==None:
        result_list,match_text_list=key_to_value.first_compare(imformation_list,image_path)
        return result_list,match_text_list
    else:
        result_list,match_text_list=key_to_value.normal_compare(imformation_list,config)
        return result_list,match_text_list