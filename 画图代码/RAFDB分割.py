'''
1	Surprise
2	Fear
3	Disgust
4	Happiness
5	Sadness
6	Anger
7   Neutral
'''
import  os
import shutil
typelist = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']
label_list_txt_path = '.\\list_patition_label.txt'
# def mkdir(out1,list1):
#     for i in list1:
#         os.makedirs(os.path.join(out1,str(i)))
#     print("ok...")
#
# a = "D:\\code_file\\sukang_one\\paper_code\\outputdataset"
# mkdir(a,typelist)


# a = 'train_00013.jpg 1'
# model_mode = a.split(" ")[0].split("_")[0]   train
# index = eval(a.split(" ")[-1])   1
# print(model_mode)
# print(typelist[index-1])
#aligned
input_path = 'D:\\code_file\\sukang_one\\paper_code\\inputdataset\\aligned'
output_path = 'D:\\code_file\\sukang_one\\paper_code\\outputdataset'
with open(label_list_txt_path, 'r') as f:
    informations = f.readlines()
    number = 1
    for i in informations:
        model_mode = i.split(" ")[0].split("_")[0]
        index = eval(i.split(" ")[-1])
        old_name = os.path.join(input_path,str(i.split(" ")[0]))
        if model_mode == 'train':
            new_name = os.path.join(output_path,str(model_mode)+"\\"+str(typelist[index-1])+"\\"+str(i.split(" ")[0]))
        if model_mode == 'test':
            new_name = os.path.join(output_path,str(model_mode)+"\\"+str(typelist[index-1])+"\\"+str(i.split(" ")[0]))
        shutil.copy(old_name,new_name)
        print("处理第%d张..."%number)
        number = number + 1
    print("处理结束...")
