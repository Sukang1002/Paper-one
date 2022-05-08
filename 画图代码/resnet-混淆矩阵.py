from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import torchvision
from torch.nn import init
import math
from torchvision import transforms
import torch.nn.functional as F
val_path = "./RAF-DB/test"
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
batch_size = 256
val_data = torchvision.datasets.ImageFolder(val_path, transform=val_transform)
data1_val = DataLoader(val_data, batch_size=batch_size, shuffle=True,drop_last=True)

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    # plt.savefig(savename, format='png')
    plt.show()

# classes表示不同类别的名称，比如这有6个类别
classes = ['Anger', 'Disgust', 'Fear', 'Happiness','Neutral', 'Sadness','Surprise']

from torchvision import models
resnet = models.resnet18()
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=3)
        self.sigmoid=nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        avp_result =self.gap(x) #bs,c,1,1
        max_result = self.maxpool(x)
        
        avp_result=avp_result.squeeze(-1).permute(0,2,1) #bs,1,c
        max_result=max_result.squeeze(-1).permute(0,2,1) #bs,1,c
        
        avp_result=self.conv(avp_result) #bs,1,c
        max_result=self.conv(max_result) #bs,1,c
        
        y=self.sigmoid(max_result + avp_result) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=3)
        self.sigmoid=nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return x*output
    
class CA_SA(nn.Module):
    def __init__(self):
        super().__init__()
        self.eca = ECAAttention()
        self.sa = SpatialAttention()
        
    def forward(self,x):
        x = self.eca(x)
        x = self.sa(x)
        return x
    
class K_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.eca = ECAAttention()
        self.sa = SpatialAttention()
        self.eca_sa = CA_SA()
    def forward(self,x):
        out1 = self.eca_sa(x)
        out2 = self.eca(x) + self.sa(x)
        return out2 + out1

class SKNet(nn.Module):
    def __init__(self, num_class=7):
        super(SKNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.KA = K_Attention()
        self.fc = nn.Linear(512*4, num_class)

    def forward(self, x):
        
        x = self.features(x)
        out1 = self.KA(x)
        out2 = self.KA(x)
        out3 = self.KA(x)
        out4 = self.KA(x)
        out = torch.cat((out1,out2,out3,out4),dim=1)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        
        return out

model_path = "./OneModel/迁移学习/ResNet18/MCML-ISA-并/qyxx-resnet18-KA-MCML.pkl"
sknet = SKNet()
checkpoint = torch.load(model_path,map_location='cpu')
sknet.load_state_dict(checkpoint['model'])
del checkpoint
def evalute_(model,val_loader):
    model.eval()

    for batchidx, (x, label) in enumerate(val_loader):
        with torch.no_grad():
            print(batchidx)
            y1 = model(x)
            _, preds1 = torch.max(F.softmax(y1,dim=1), 1)
            if batchidx!=0:
                y = torch.cat((y,preds1),dim=0)
                labels = torch.cat((labels,label),dim=0)
            else:
                y = preds1
                labels = label
    print(y.shape)
    print(labels.shape)
    assert y.shape == labels.shape
    y = y.numpy()
    
    return y,labels

y_pred,y_true = evalute_(model=sknet,val_loader=data1_val)


# 获取混淆矩阵
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, './confusion_matrix.png', title='confusion matrix')