# -*- coding: utf-8 -*-
# @Time    : 2021/3/2 下午 2:56
# @Author  : 划水小苏
# @FileName: 测试数据集选取.py

import shutil
import os

'''
anger 135，disgust 177，fear 75，happy 207，sadness 84，surprise 249  981
anger 27， disgust 36，fear 15，happy 42，sadness 17，surprise 50
kinds_search_numbers = [135,177,75,207,84,249] ,kinds_search_numbers
shutil.move('原文件夹/原文件名','目标文件夹/目标文件名')
 ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
'''

#创建输出文件夹
def test_mkdir(outputpath,class_kinds):

    for i in class_kinds:
        folder = os.path.exists(os.path.join(outputpath, i))
        if not folder:
            os.mkdir(os.path.join(outputpath,i))
        else:
            print(i,'已经存在了。')
def test_data_search(datasets_path,data_test_list,output_path):
    class_kinds = os.listdir(datasets_path)
    for i in class_kinds:
        #获取每类测试值
        number = data_test_list[i]
        for k in range(number):
            shutil.move(os.path.join(datasets_path, i+'\\'+str(i)+'_'+str(k)+'.png'),
            os.path.join(output_path,i+'\\'+str(i)+'_'+str(k)+'.png'))


data_test_list = {'anger': 27, 'disgust': 36, 'fear': 15, 'happy': 42, 'sadness': 17, 'surprise': 50}
datasets_path = './test'
output_path = './CK/test'
class_kinds = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

test_mkdir(output_path,class_kinds)
test_data_search(datasets_path,data_test_list,output_path)