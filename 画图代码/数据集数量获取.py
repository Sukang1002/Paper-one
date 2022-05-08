import os
def Rename(input_path):
    sum = 0
    kinds_name_list = os.listdir(input_path)
    for i in kinds_name_list:
        kinds_name_path = os.path.join(input_path,i)
        number = 0
        image_name_list = os.listdir(kinds_name_path)
        for k in image_name_list:
            number = number + 1
        print("%s处理了%i张"%(i,number))
        sum +=number
    print(sum)
    return 1

path = 'D:\\code_file\\sukang_one\\MTCNN-PyTorch\\inputdataset\\val'
Rename(path)