#关于CK+数据集的增强
import os
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

def Aug(input_path,output_path):
    ChannelShuffle = iaa.ChannelShuffle(0.35, channels=[0, 1]) #ChannelShuffle
    AddElementwise = iaa.AddElementwise((-40, 40)) #将值添加到图像的像素中，相邻像素可能具有不同的值。
    Multiply = iaa.Multiply((0.5, 1.5), per_channel=0.5)#将所有图像的 50% 与 0.5 和 1.5 之间的随机值相乘，然后将剩余的 50% 逐个通道相乘，即每个通道独立采样一个乘数
    Cutout = iaa.Cutout(nb_iterations=2) #每个图像填充两个随机区域，默认为灰色像素

    CoarseDropout = iaa.CoarseDropout(0.02, size_percent=0.5)#通过将所有像素转换为黑色像素来丢弃 2% 的像素，但在具有原始大小 50% 的图像的较低分辨率版本上执行此操作，会导致 2x2 正方形被丢弃
    Dropout2d = iaa.Dropout2d(p=0.6)#平均丢弃所有图像通道的一半。丢弃的通道将用零填充。每个图像中至少有一个通道保持不变（默认设置）
    SaltAndPepper = iaa.SaltAndPepper(0.1, per_channel=True)#用椒盐噪声 替换所有像素的通道： 10%
    GaussianNoise = iaa.imgcorruptlike.GaussianNoise(severity=2)#高斯噪音 等级[1,5]

    Brightness = iaa.imgcorruptlike.Brightness(severity=2)#亮度
    Snow = iaa.imgcorruptlike.Snow(severity=1)#加雪
    Fliplr = iaa.Fliplr(1)#水平翻转
    GammaContrast = iaa.GammaContrast((0.5, 2.0), per_channel=True)#伽玛对比度
    #获取类别列表
    kinds_name_list = os.listdir(input_path)
    #迭代进入每一个类别
    for i in kinds_name_list:
        kinds_name_path = os.path.join(input_path,i)
        number = 0
        #判断输出路径是否构成类别文件夹
        folder = os.path.exists(os.path.join(output_path, i))
        if not folder:
            os.mkdir(os.path.join(output_path,i))
        else:
            print(i,'已经存在了。')
        #获取每类中的每一张图片，构成列表
        image_name_list = os.listdir(kinds_name_path)
        for k in image_name_list:
            #获取原图路径
            image_path = os.path.join(kinds_name_path,k)
            #加载图片
            image = imageio.imread(image_path)
            #构建路径
            o_image_yuan_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_yuan"+".png")
            o_image_cs_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_cs"+".png")
            o_image_ae_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_ae"+".png")
            o_image_mu_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_mu"+".png")
            o_image_c_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_c"+".png")
            o_image_cd_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_cd"+".png")
            o_image_d2_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_d2"+".png")
            o_image_sp_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_sp"+".png")
            o_image_gn_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_gn"+".png")
            o_image_b_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_b"+".png")
            o_image_s_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_s"+".png")
            o_image_f_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_f"+".png")
            o_image_g_path = os.path.join(os.path.join(output_path, i),str(i)+"_"+str(number)+"_g"+".png")
            #数据增强
            one = ChannelShuffle(image=image)
            two = AddElementwise(image=image)
            three = Multiply(image=image)
            four = Cutout(image=image)

            five = CoarseDropout(image=image)
            six = Dropout2d(image=image)
            seven = SaltAndPepper(image=image)
            eight = GaussianNoise(image=image)

            nine = Brightness(image=image)
            ten = Snow(image=image)
            eleven = Fliplr(image=image)
            twelve = GammaContrast(image=image)
            #保存图片
            imageio.imsave(o_image_yuan_path,image)
            imageio.imsave(o_image_cs_path,one)
            imageio.imsave(o_image_ae_path,two)
            imageio.imsave(o_image_mu_path,three)
            imageio.imsave(o_image_c_path,four)
            imageio.imsave(o_image_cd_path,five)
            imageio.imsave(o_image_d2_path,six)
            imageio.imsave(o_image_sp_path,seven)
            imageio.imsave(o_image_gn_path,eight)
            imageio.imsave(o_image_b_path,nine)
            imageio.imsave(o_image_s_path,ten)
            imageio.imsave(o_image_f_path,eleven)
            imageio.imsave(o_image_g_path,twelve)
            number = number + 1
        print("%s处理了%i张"%(i,number))


input_path = "C:\\Users\\23096\\Desktop\\python_code\\RAF-DB\\train"
output_path = "C:\\Users\\23096\\Desktop\\python_code\\out"
Aug(input_path,output_path)