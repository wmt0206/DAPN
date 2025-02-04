import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from PIL import ImageEnhance
import numpy as np
import random
from PIL import ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
identity = lambda x:x
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]
        
    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


# 高斯模糊
class PILRandomGaussianBlur(object):

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    # 使用numpy生成一个[0, 1) 之间的随机数，判断是否小于等于self.prob，决定是否应用高斯模糊。
    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img


        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
            
# 颜色失真
def get_color_distortion(s=0.5):
    # s is the strength of color distortion.
    # 颜色抖动:应用随机的亮度、对比度、饱和度和色调变化到图像上
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    # 颜色抖动变换将以 80% 的概率应用到图像上
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    # 随机灰度化:以20%的概率将图像转换为灰度图像。它帮助模型学习不依赖颜色信息的特征。
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    # 组合变换
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

# 根据类别组织数据，并为每个类别创建一个单独的数据加载器
# 方便在训练过程中按episode处理数据
class SetDataset:
    def __init__(self, data_path, num_class, batch_size):
        self.sub_meta = {}
        self.data_path = data_path
        self.num_class = num_class
        self.cl_list = range(self.num_class)
        # 遍历所有类别，并为每个类别在self.sub_meta字典中创建一个空列表，准备存放该类别的数据样本。
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        # 该类会根据目录结构自动分配类别标签。
        d = ImageFolder(self.data_path)
        # print('-------------------------------------------------')
        # print(d)
        # print('-------------------------------------------------')
        for i, (data, label) in enumerate(d):
            # print('-------------------------------------------------')
            # print(label)
            # print('-------------------------------------------------')
            self.sub_meta[label].append(data)
        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))

        # 初始化一个空列表，用于存放每个类别的数据加载器。
        # pin_memory：是否将数据复制到CUDA可访问的内存中，设为False。
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl])
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

# 主要用于数据增强和预处理
class SubDataset:
    def __init__(self, 
        sub_meta, 
        size_crops=[224, 96],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14],
    ):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 这循环为每个尺寸创建一个变换列表(trans)，每个列表包含多个transforms.Compose对象，每个对象定义了应用于图像的一系列变换步骤：
        # 随机缩放裁剪、随机水平翻转、颜色失真、转为Tensor以及标准化。
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        # 定义全局图像抖动参数，用于调整亮度、对比度和色彩。
        self.jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)

        # 应用于整个图像的全局变换，包括固定尺寸的调整、图像抖动、水平翻转、转Tensor以及归一化。
        self.global_transforms = transforms.Compose([
                transforms.Resize([224,224]),
                ImageJitter(self.jitter_param),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        
        self.sub_meta = sub_meta

    # 数据获取方法
    # 对于给定的索引i，获取相应的图像，并对该图像应用之前定义的所有局部变换，生成多个裁剪视图，同时应用全局变换并将其加入裁剪视图列表，最后返回这个包含多个视图的列表。
    def __getitem__(self,i):
        
        img = self.sub_meta[i] 
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        raw_image = self.global_transforms(img)
        multi_crops.append(raw_image)
        
        return multi_crops


    def __len__(self):
        return len(self.sub_meta)
    
# 自定义的数据采样器，用于生成 episode
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    # 为每个episode生成类别的索引
    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class Eposide_DataManager():
    def __init__(self, data_path, num_class, n_way=5, n_support=1, n_query=15, n_eposide=1):        
        super(Eposide_DataManager, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

    # 生成数据加载器
    def get_data_loader(self): 
        dataset = SetDataset(self.data_path, self.num_class, self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)

        # print('--------------------------------')
        # print("Episode categories:")
        # for episode_index in sampler:
        #     print(episode_index)
        # print('--------------------------------')

        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)   
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    
    np.random.seed(1111)
    data_path = "./source_domain/miniImageNet/train"
    datamgr = Eposide_DataManager(data_path=data_path, num_class=64, n_way=2, n_support=1, n_query=2, n_eposide=2)
    base_loader = datamgr.get_data_loader()
    data = []
    for i, x in enumerate(base_loader):
        x_96 = torch.stack(x[2:8]) # (6,way,shot+query,3,96,96)
        print(x_96.shape)
        x_224 = torch.stack(x[8:])  # (1,way,shot+query,3,224,224)
        print(x_224.shape)
    
    