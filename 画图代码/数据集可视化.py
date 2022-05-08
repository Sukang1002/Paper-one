# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 上午 10:06
# @Author  : 划水小苏
# @FileName: 数据集可视化.py
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import numpy as np
#设置中文表达
mpl.rcParams["font.sans-serif"]=["SimHei"]


# #随机选取6张图片可视化
# class_idx = {0:'anger',1:'disgust',2:'fear',3:'happy',4:'sadness',5:'surprise'}
# transform = transforms.Compose([
#                                 transforms.ToTensor()])
# #transforms.RandomHorizontalFlip(p=0.5)
# train_path = "./test"
# train_data = ImageFolder(train_path, transform=transform)
# data0_train = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

# examples = enumerate(data0_train)
# batch_idx, (example_data, example_targets,image_path) = next(examples)
# print(example_targets)
# print(example_data.shape)

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   index = str(example_targets[i].numpy())
#   plt.title("Ground Truth: {}".format(class_idx[eval(index)]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()


#数据总体可视化
# def imagesclass_show(dataset_path):
#     a = {}
#     file_exist = os.path.isfile(dataset_path)
#     if not file_exist:
#         sort_list = os.listdir(dataset_path)
#         for i in sort_list:
#             sort_path_list  = os.path.join(dataset_path,i)
#             images_list = os.listdir(sort_path_list)
#             # print(i,"类别有 : ",len(images_list))
#             a[str(i)] = len(images_list)
#         return a
#     else:
#         print("not found file%s"%(dataset_path))

# # b = imagesclass_show("./test")
# b = {'anger': 108, 'disgust': 141, 'fear': 60, 'happy': 165, 'sadness': 67, 'surprise': 199}

# print(b.keys())

# sort_name = []
# values = []
# for i in b.keys():
#     sort_name.append(i)
#     values.append(b[i])
# print(sort_name)
# print(values)

# def image_plot(sort_names,numbers,save = False,datasets_name="Datasets",x_name="图片种类",y_name="图片数量/张",N=6):
#     max_number = max(numbers) + 30
#     # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
#     plt.figure(figsize=(8, 6), dpi=80)
#     # 再创建一个规格为 1 x 1 的子图
#     plt.subplot(1, 1, 1)
#     # 柱子总数
#     # 包含每个柱子对应值的序列
#     # 包含每个柱子下标的序列
#     index = np.arange(N)
#     # 柱子的宽度
#     width = 0.35
#     # 绘制柱状图, 每根柱子的颜色为紫罗兰色
#     p2 = plt.bar(index, numbers, width, label='图片数量', color="#87CEFA")
#     # 设置横轴标签
#     plt.xlabel(x_name)
#     # 设置纵轴标签
#     plt.ylabel(y_name)
#     # 添加标题
#     plt.title(datasets_name)
#     # 添加纵横轴的刻度
#     plt.xticks(index, sort_name)
#     plt.yticks(np.arange(0, max_number, 20))
#     # 添加图例
#     plt.legend(loc="upper right")
#     if save:
#         plt.savefig("./test%s.png"%(datasets_name))
#     plt.show()
#     return 1

# image_plot(sort_name,values,save=True)


def image_compare(image_path_list,name_list):
    # plt.figure()  #figsize=(10,4)
    for i in range(len(image_path_list)):
        exec("plt.subplot(1, 7, {})".format(i+1))
        exec("plt.title(str(name_list[{}]))".format(i))
        exec("plt.imshow(cv2.imread(image_path_list[{}]))".format(i))
    plt.show()
    return 1


#数据集增强可视化 plt.subplot(3, 4, {})
# image_path_list = ['./out/Anger/Anger_0_ae.png','./out/Anger/Anger_0_b.png','./out/Anger/Anger_0_c.png' ,\
#     './out/Anger/Anger_0_cd.png','./out/Anger/Anger_0_cs.png','./out/Anger/Anger_0_d2.png',\
#         './out/Anger/Anger_0_f.png','./out/Anger/Anger_0_g.png','./out/Anger/Anger_0_gn.png',\
#             './out/Anger/Anger_0_mu.png','./out/Anger/Anger_0_s.png','./out/Anger/Anger_0_sp.png'
#     ]
# name_list = ['ae','b','c','cd','cs','d2','f','g','gn','mu','s','sp']
# image_compare(image_path_list,name_list)


# image_path_list = ['./CK+48/anger/anger_yuan_0.png','./CK+48/contempt/contempt_yuan_0.png','./CK+48/disgust/disgust_yuan_0.png',\
#     './CK+48/fear/fear_yuan_0.png','./CK+48/happy/happy_yuan_0.png','./CK+48/sadness/sadness_yuan_0.png',\
#         './CK+48/surprise/surprise_yuan_0.png'
#     ]
# name_list = ['anger', 'contempt','disgust', 'fear', 'happy', 'sadness', 'surprise']
# image_compare(image_path_list,name_list)

image_path_list = ['./RAF-DB/train/Anger/Anger_yuan_0.png','./RAF-DB/train/Disgust/Disgust_yuan_0.png','./RAF-DB/train/Fear/Fear_yuan_0.png',\
    './RAF-DB/train/Happiness/Happiness_yuan_0.png','./RAF-DB/train/Neutral/Neutral_yuan_0.png','./RAF-DB/train/Sadness/Sadness_yuan_0.png',\
        './RAF-DB/train/Surprise/Surprise_yuan_0.png'
    ]
name_list = ['anger','disgust', 'fear', 'happy','neutral', 'sadness', 'surprise']
image_compare(image_path_list,name_list)