import pandas as pd

def Search(csv_path_list,key):
    for i in range(len(csv_path_list)):
        print()
        exec('data{} = pd.read_csv(csv_path_list[i])'.format( i ))
        exec('print(data{}[key].max())'.format( i ))
    return 1

path1 = './data_qx_kanet.csv'  #0.8696
path2 = "./data_qx_mcml.csv"  #0.8709
# path3 = './数据清洗后/6/data_qx_base-qy.csv'  #0.8784
# path4 = './数据清洗后/6/data_qx_base.csv'  #0.8784
List = [path1,path2]
key = 'val_acc'
Search(List,key=key)