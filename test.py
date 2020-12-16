# coding:utf8
import matplotlib.pyplot as plt
from torch.types import Number
from torch.utils import data
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix,precision_score
import torch


def visualize(data, preds, labels):
    plt.figure
    # print(data.size()) #一开始的大小为torch.Size([4, 3, 224, 224])
    out = make_grid(data) #这样得到的输出的数据就是将四张图合成了一张图的大小，为
    # print(out.size()) #torch.Size([3, 228, 906])
    #因为反标准化时需要将图片的维度从（channels,imgsize,imgsieze)变成（imgsize,imgsieze,channels)，这样才能与下面的std,mean正确计算
    inp = torch.transpose(out, 0, 2)
    # print(inp.size()) #返回torch.Size([906, 228, 3])
    mean = torch.cuda.FloatTensor([0.485, 0.456, 0.406])
    std = torch.cuda.FloatTensor([0.229, 0.224, 0.225])
    inp = std * inp + mean
    #计算完后还是要将维度变回来，所以再进行一次转换
    inp = torch.transpose(inp, 0, 1)
    # print(inp.size()) #返回torch.Size([3, 228, 906])

    #注意，这里是因为设置了batch_size为四，所以title中才这样，如果你的batch_size不是4这里需要做一些更改
    plt.imshow(inp.cpu())
    plt.title('get {},{},{},{} for {},{},{},{}'.format(preds[0].item(), preds[1].item(), preds[2].item(), preds[3].item(),labels[0],labels[1],labels[2],labels[3]))
    plt.show()
    #比如下面这个就是将batch_size改成1的结果
    # viz.images(inp, opts=dict(title='{}'.format(preds[0].item())))

def self_dataset():
    data_test_root = 'C:\\Users\\Mia\\Downloads\\ddnl\\imgs\\test' #测试数据集所在的路径
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    datatransform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                ])
    test_data = datasets.ImageFolder(data_test_root,datatransform)    #如果只测试一张图片，这里batch_size要改成1
    dataloaders = data.DataLoader(test_data, batch_size=4 ,shuffle=True,num_workers=2)
    all_labels = []
    all_preds = []
    for inputs,labels in dataloaders:
        inputs = inputs.to(device)   # 当前批次的训练输入 
        outputs = model_test(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        #visualize(inputs,preds,labels)
    return all_labels, all_preds


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = 'RES'
    accuracy = []
    recall = []
    precision = []
    con_mat =[]
    for k in np.arange(1,4):
        #导入上面训练得到的效果最佳的网络，因为测试是在只有cpu的机器上跑的，所以这里要添加map_location='cpu'
        model_test = torch.load('./Gender'+model+'_adam_{}.pkl'.format(k))
        #如果你是在有GPU的机器上跑的，可以删掉map_location='cpu'，并添加一行
        model_test = model_test.to(device)
        model_test.eval()
        
        labels, preds = self_dataset()
        accuracy.append(accuracy_score(labels, preds))
        recall.append(recall_score(labels, preds))
        con_mat=confusion_matrix(labels,preds)
        precision.append(precision_score(labels, preds))
        print('Confusion matrix (learning rate = {}) : \n'.format(1e-2**k),con_mat)

    plt.figure
    plt.plot(np.arange(1,4),accuracy,np.arange(1,4),recall,np.arange(1,4),precision)
    plt.legend(['accuracy','recall','precision'])
    plt.title(model+'Net')
    plt.xlabel('-2 * Log of learning rate')
    plt.show()
    