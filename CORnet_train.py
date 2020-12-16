# coding:utf8
from torchvision import datasets
from torch import nn, optim
from torchvision import transforms as T
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

import os
import copy
import time
import torch
import cornet

model_type = 'S' # choices=['Z', 'R', 'RT', 'S']
times = 5

#首先进行数据的处理
data_dir = 'C:\\Users\\Mia\\Downloads\\ddnl\\imgs'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#转换图片数据
normalize = T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
data_transforms ={
    'train': T.Compose([
        T.RandomResizedCrop(224),#从图片中心截取
                T.RandomHorizontalFlip(),#随机水平翻转给定的PIL.Image,翻转概率为0.5  
        T.ToTensor(),#转成Tensor格式，大小范围为[0,1]
        normalize
    ]),

    'val': T.Compose([
        T.Resize(256),#重新设定大小 
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ]),
}

#加载图片
#man的label为0, woman的label为1
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

#得到train和val中的数据量
dataset_sizes = {x : len(image_datasets[x].imgs) for x in ['train', 'val']}
dataloaders = {x : data.DataLoader(image_datasets[x], batch_size=4, shuffle=True,num_workers=4) for x in ['train', 'val']}

def get_model(pretrained=False):
    #选择使用的模型
    map_location = None 
    model = getattr(cornet, f'cornet_{model_type.lower()}')
    if model_type.lower() == 'r':
        model = model(pretrained=pretrained, map_location=map_location, times=times)
    else:
        model = model(pretrained=pretrained, map_location=map_location)
    #修改类别为2,即man和woman
    model.module.decoder.linear.out_features = 2
    model.to(device)
    return model


# 训练模型
# 参数说明：
# model:待训练的模型
# criterion：评价函数
# optimizer：优化器
# scheduler：学习率
# num_epochs：表示实现完整训练的次数，一个epoch表示一整個训练周期
def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    # 定义训练开始时间
    since = time.time()
    #用于保存最优的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    #最优精度值
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_iteration = 0
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    # # meters,统计指标：平滑处理之后的损失，还有混淆矩阵
    # loss_meter = meter.AverageValueMeter()#能够计算所有数的平均值和标准差，用来统计一个epoch中损失的平均值
    # confusion_matrix = meter.ConfusionMeter(2)#用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标

    # 对整个数据集进行num_epochs次训练
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        #用于存储train acc还没有与val acc比较之前的值
        temp = 0
        # Each epoch has a training and validation phase
        # 每轮训练训练包含`train`和`val`的数据
        for phase in ['train', 'val']:
            if phase == 'train':
                # 学习率步进
                scheduler.step()
                # 设置模型的模式为训练模式（因为在预测模式下，采用了`Dropout`方法的模型会关闭部分神经元）
                model.train()  # Set model to training mode
            else:
            # 预测模式
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # 遍历数据，这里的`dataloaders`近似于一个迭代器，每一次迭代都生成一批`inputs`和`labels`数据,
            # 一批有四个图片，一共有dataset_sizes['train']/4或dataset_sizes['val']/4批
            # 这里循环几次就看有几批数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)   # 当前批次的训练输入 
                labels = labels.to(device)  # 当前批次的标签输入
                # print('input : ', inputs)
                # print('labels : ', labels)

                # 将梯度参数归0
                optimizer.zero_grad()

                # 前向计算
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 相应输入对应的输出
                    outputs = model(inputs)
                    # print('outputs : ', outputs)
                    # 取输出的最大值作为预测值preds,dim=1,得到每行中的最大值的位置索引，用来判别其为0或1
                    _, preds = torch.max(outputs, 1)
                    # print('preds : ', preds)
                    # 计算预测的输出与实际的标签之间的误差
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                    # 对误差进行反向传播
                        loss.backward()
                        #scheduler.step(loss) #当使用的学习率递减函数为optim.lr_scheduler.ReduceLROnPlateau时，使用在这里
                        # 执行优化器对梯度进行优化
                        optimizer.step()

                        # loss_meter.add(loss.item())
                        # confusion_matrix.add(outputs.detach(), labels.detach()) 

                # statistics
                # 计算`running_loss`和`running_corrects`
                #loss.item()得到的是此时损失loss的值
                #inputs.size(0)得到的是一批图片的数量，这里为4
                #两者相乘得到的是4张图片的总损失
                #叠加得到所有数据的损失
                running_loss += loss.item() * inputs.size(0)
                #torch.sum(preds == labels.data)判断得到的结果中有几个正确，running_corrects得到四个中正确的个数
                #叠加得到所有数据中判断成功的个数
                running_corrects += torch.sum(preds == labels.data)

        # 当前轮的损失,除以所有数据量个数得到平均loss值
            epoch_loss = running_loss / dataset_sizes[phase]
            # 当前轮的精度，除以所有数据量个数得到平均准确度
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # 对模型进行深度复制 
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            if phase == 'val':   
                val_acc.append(epoch_acc) 
                val_loss.append(epoch_loss)    
            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
            if phase =='val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_iteration = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    # 计算训练所需要的总时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:4f}'.format(best_iteration))  
    print('Best train Acc: {:4f}'.format(best_train_acc)) 
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # load best model weights
    # 加载模型的最优权重
    model.load_state_dict(best_model_wts)
    return model,train_acc,val_acc,train_loss,val_loss



if __name__ == '__main__':
    epochs = 100
    for idx,learningrate in enumerate(0.01**np.arange(1,4)):
        model = get_model(pretrained=True)
        #定义使用的损失函数为交叉熵代价函数
        criterion = nn.CrossEntropyLoss()
        #定义使用的优化器
        #optimizer_conv = optim.SGD(model_conv.parameters(),lr=0.0001,momentum=.9,weight_decay=1e-4)
        #optimizer_conv = optim.RMSprop(model_conv.parameters(),lr = 0.0001, momentum=.9,weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=learningrate, betas=(0.9, 0.99))
        #设置自动递减的学习率,等间隔调整学习率,即在7个step时，将学习率调整为 lr*gamma
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
        #exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode='min', verbose=True)
        model_train,train_acc,val_acc,train_loss,val_loss = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=epochs)
        plt.figure
        plt.plot(np.arange(1,epochs+1),train_acc,np.arange(1,epochs+1),val_acc)
        plt.legend(['train accuracy','validation accuracy'])
        plt.title('CORNet accuracy')
        plt.xlabel('Number of epochs')
        plt.show()
        plt.figure
        plt.plot(np.arange(1,epochs+1),train_loss,np.arange(1,epochs+1),val_loss)
        plt.legend(['train loss','validation loss'])
        plt.title('CORNet loss')
        plt.xlabel('Number of epochs')
        plt.show()
        torch.save(model_train, 'GenderCOR_adam_{}.pkl'.format(idx+1))