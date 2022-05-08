from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import torchvision
from torch.nn import init
from torchvision import transforms
val_path = "./RAF-DB/test"
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
batch_size = 128
val_data = torchvision.datasets.ImageFolder(val_path, transform=val_transform)
data1_val = DataLoader(val_data, batch_size=batch_size, shuffle=True,drop_last=True)

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

resnet = models.resnet18()
class SKNet(nn.Module):
    def __init__(self, num_class=7):
        super(SKNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        
        x = self.features(x)
        out = self.avgpool(x)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


class KNet(nn.Module):
    def __init__(self, num_class=7):
        super(KNet, self).__init__()
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
model1 = SKNet()
model2 = KNet()
model_path1 = "./OneModel/迁移学习/ResNet18/Base/qyxx-resnet18.pkl"
checkpoint1 = torch.load(model_path1,map_location='cpu')
model1.load_state_dict(checkpoint1['model'])
del checkpoint1

model_path2 = "./OneModel/迁移学习/ResNet18/MCML-ISA-并/qyxx-resnet18-KA-MCML.pkl"
checkpoint2 = torch.load(model_path2,map_location='cpu')
model2.load_state_dict(checkpoint2['model'])
del checkpoint2

def evalute_(model1,model2,val_loader):
    model1.eval()
    model2.eval()
    for batchidx, (x, label) in enumerate(val_loader):
        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)
            break
    y1 = y1.numpy()
    y2 = y2.numpy()
    return y1,y2,label

x1,x2, y1 = evalute_(model1=model1,model2=model2,val_loader=data1_val)


tar = ['anger','disgust','fear','happy','neutral','sadness','surprise']
X_norm1 = TSNE(n_components=2, learning_rate='auto',
                  init='pca').fit_transform(x1)

X_norm2 = TSNE(n_components=2, learning_rate='auto',
                  init='pca').fit_transform(x2)
x_min, x_max = X_norm1.min(0), X_norm1.max(0)
X_norm_1 = (X_norm1 - x_min) / (x_max - x_min)  # 归一化

x_min, x_max = X_norm2.min(0), X_norm2.max(0)
X_norm_2 = (X_norm2 - x_min) / (x_max - x_min)  # 归一化

plt.figure(figsize=(8, 8))
for i in range(X_norm_1.shape[0]):
    plt.text(X_norm_1[i, 0], X_norm_1[i, 1], tar[y1[i].item()], color=plt.cm.Set1(y1[i]), 
             fontdict={'weight': 'bold', 'size': 9})

plt.figure(figsize=(8, 8))
for i in range(X_norm_2.shape[0]):
    plt.text(X_norm_2[i, 0], X_norm_2[i, 1], tar[y1[i].item()], color=plt.cm.Set1(y1[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()