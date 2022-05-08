# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 下午 7:59
# @Author  : 划水小苏
# @FileName: 文件命名.py

#文件重命名
'''
代码复用要求
文件夹路径如下所示
master_path
    |__class1
        |__class1_1
        |__class1_2
        |__class1_3
        ...
    |__class2
        |__class2_1
        |__class2_2
        |__class2_3
        ...
    |__class3
        |__class3_1
        |__class3_2
        |__class3_3
        ...
'''

#命名格式 classi_yuan_number.png
import os
def Rename(input_path):
    kinds_name_list = os.listdir(input_path)
    for i in kinds_name_list:
        kinds_name_path = os.path.join(input_path,i)
        number = 0
        image_name_list = os.listdir(kinds_name_path)
        for k in image_name_list:
            old_image_path = os.path.join(kinds_name_path,k)
            os.rename(old_image_path,os.path.join(kinds_name_path,i+'_yuan_'+str(number)+".png"))
            number = number + 1
        print("%s处理了%i张"%(i,number))
    return 1


#CK+
# path = "C:\\Users\\23096\\Desktop\\python_code\\CK+48"
# Rename(path)
# #anger:135,contempt:177,disgust:177,fear:75,happy:207,sadness:84,surprise:249 

#RAF_DB-train
# path = "C:\\Users\\23096\\Desktop\\python_code\\RAF-DB\\train"
# Rename(path)
# #anger:705,disgust:717,fear:281,happy:4772,sadness:1982,surprise:1290,Neutral:2524 

#RAF_DB-test
# path = "C:\\Users\\23096\\Desktop\\python_code\\RAF-DB\\test"
# Rename(path)
# #anger:162,disgust:160,fear:74,happy:1185,sadness:680,surprise:478,Neutral:329 