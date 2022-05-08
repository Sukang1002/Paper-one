from os import path
from numpy import testing
import torch
print("hello")
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as mb
import matplotlib.pyplot as plt
'''
感叹号 “!” 注意格式。
问号“?” 代码有错。
TODO 代码未来将要进行的操作。
@param 参数
'''
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



#数据清洗
#!csv_path = "csv文件所处路径"    key = '区别其他csv文件的标志'
def Data_recovery(csv_path , key,epoch):
    xls = pd.read_csv(csv_path , names=['epoch','train_acc','train_loss','val_acc','val_loss'])
    # print(xls['train_loss'][0])
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    for i in range(epoch):
        train_acc_list.append(eval((xls['train_acc'][i].split(',')[0]).split('(')[-1]))
        train_loss_list.append(xls['train_loss'][i])
        test_acc_list.append(eval((xls['val_acc'][i].split(',')[0]).split('(')[-1]))
        test_loss_list.append(xls['val_loss'][i])
    # 清洗数据保存
    save_path = "./data_qx_%s.csv"%(str(key))
    csv_ = pd.DataFrame({'train_acc':train_acc_list,'train_loss':train_loss_list,'val_acc':test_acc_list,'val_loss':test_loss_list})
    csv_.to_csv(save_path , index=False,sep=',')
    return save_path

#数据读取进行画图
# !other_list = ['x_name','y_name','title_name',epoch]
# !key = "val_acc" || "train_acc"
# !label_list = ["为每条线条代表的意思", ]
# !csv_path_list = [ "csv文件所处的位置"，]
def Picture(csv_path_list , label_list , key , other_list):
    c = ['r','yellow','black','deepskyblue','green','blueviolet']
    for i in range(len(csv_path_list)):
        exec('data{} = pd.read_csv(csv_path_list[i])'.format( i ))
        
    epoch_list = []
    for i in range(len(csv_path_list)):
        exec('test_acc_list{} = {}'.format( i , []))
    #将csv文件中的数据添加到列表里面（提供y轴数量标）
    for j in range(other_list[-1]):
        for i in range(len(csv_path_list)):
            exec('test_acc_list{}.append(data{}[key][j])'.format( i , i ))
    #创建x轴数量标
    for i in range(other_list[-1]):
        epoch_list.append(i)
    #粗略计算数组不等情况
    # 多条可再增加plt.plot进行划分，利用颜色区分准确率
    
    for i in range(len(csv_path_list)):
        # exec('print(test_acc_list{})'.format(i))
        exec("plt.plot(epoch_list, test_acc_list{}, color = c[i] , label = label_list[i])".format(i))
    plt.xlabel(other_list[0])  # x轴表示
    plt.ylabel(other_list[1])  # y轴表示
    plt.title(other_list[2])  # 图标标题表示
    plt.legend()
    plt.savefig('./test.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片

#判断模型是否过拟合
#需求分析，只需要同一模型的训练误差和测试误差
# !other_list = ['x_name','y_name','title_name',epoch]
# !csv_path_list = [ "csv文件所处的位置"，]
def Overfitting(path , other_list):
    c = ['r','black']
    path1 = pd.read_csv(path)
    train_loss = []
    val_loss = []
    epoch_list = []
    #将csv文件中的数据添加到列表里面（提供y轴数量标）
    for i in range(epoch):
        train_loss.append(path1['train_loss'][i])
        val_loss.append(path1['val_loss'][i])
        epoch_list.append(i)  #创建x轴数量标
    #粗略计算数组不等情况
    # 多条可再增加plt.plot进行划分，利用颜色区分准确率
    plt.plot(epoch_list,train_loss,color = c[0] , label = 'train_loss')
    plt.plot(epoch_list,val_loss,color = c[1] , label = 'val_loss')
    plt.xlabel(other_list[0])  # x轴表示
    plt.ylabel(other_list[1])  # y轴表示
    plt.title(other_list[2])  # 图标标题表示
    plt.legend()
    # plt.savefig('test.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片
        

path1 = "./OneModel/泛化能力/ecanet-位置信息/qyxx-eca_net-KA-MCML-information.csv"
path2 = "./OneModel/泛化能力/ecanet-base/qyxx-eca_net-KA-MCML-CK.csv"
path3 = "./OneModel/泛化能力/ecanet-KA-MCML/qyxx-eca_net-KA-MCML-CK.csv"
path4 = "./OneModel/泛化能力/ecanet-qy/qyxx-eca_net-KA.csv"

epoch = 196
path1_qx = Data_recovery(csv_path=path1,key="base-inm",epoch=epoch)
path2_qx = Data_recovery(csv_path=path2,key="base",epoch=epoch)
path3_qx = Data_recovery(csv_path=path3,key="base-KA-MCML",epoch=epoch)
path4_qx = Data_recovery(csv_path=path4,key="base-qy",epoch=epoch)

csv_path_list = [path1_qx,path2_qx,path3_qx,path4_qx]
label_list = ['base-inm','base','base-KA-MCML','base-qy']
key = 'val_acc'
other_list = ['epoch','val_acc','ECANet',epoch]
# Overfitting(path = path,other_list = other_list)
Picture(csv_path_list=csv_path_list,label_list=label_list,key=key,other_list=other_list)