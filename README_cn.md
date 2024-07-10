
# Spike-EEG-Network

[English](README.md) | 中文(Chinese)
## Data Prepare

请将下载的数据集按照如下的方式进行，以 [KUL Database](https://zenodo.org/records/4004271) 和 [DTU DataBase](https://zenodo.org/records/1199011) 为例：
```
|EEG Database/
├──KUL Dataset/
│  ├── Data
│  │   ├── S1.mat
│  │   ├── S2.mat
│  │   ├── ......
│  ├── ......
├──DTU Dataset/
│  ├── EEG
│  │   ├── S1.mat
│  │   ├── S2.mat
│  │   ├── ......
│  ├── Data_preproc
│  │   ├── ......
│  ├── AUDIO
│  │   ├── ......
|  ├── ......
```

### Data Loader

接下来，我们将借助 EEGLoader.py 将数据集处理，读取到的数据进行训练集和测试集的划分，以便后续网络的输入。
```
from EEGLoader import EEGDataset
path = r'' # EEG 数据的根目录

train_ratio = 0.8 ## train_set 比例
test_ratio = 0.2 ## test_ratio 比例

train_set, test_set = EEGDataset(path, train_ratio, test_ratio)
```

其中 ```train_set``` 的类型为 ```<torchvision.datasets>```。之后，我们将可以通过 ```<torch.utils.data.DataLoader>``` 的方式将处理之后的结果转换为网络可以训练的模式。

```
from torch.utils.data import Dataset,DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=True,pin_memory=True)
```

### Data Show

![](EEGShow.png)
为了帮助我们更好的理解所读取的数据的形式，我们将展示如何将读取到的EEG Data数据进行展示。
```
import matplot.pyplot as plt
import scipy.io
import mne

# input is EEG Data
EEGShow(train_set[0])

# print EEG's shape: [Batch_size, C, H, W] 
print(train_set[0].shape)

# draw with mne package
raw = mne.io.read_raw_eeglab(filename)
fig = raw.plot(title='EEG Data', show = True)
fig.savefig('result.jpg',dpi = 550)
break
```

## Spike Encoding

为了使得输入的 EEG 数据可以转换为适合 Spiking Neuron Networks 的脉冲序列的形式，我们需要设计额外的脉冲编码层，使得时序的 EEG 信号，转换为 Spike Pattern 的形式。

针对 EEG 信号的输入的形式，我们可以将 Spike Encoding 分为 频谱分析形式 以及 端到端的处理模式。

### 基于频谱分析的EEG信号分类
频谱分析的范式是借助频谱分析的范式将 EEG 信号分解成不同频带的特征信息，并借助频谱分析的一般方法，将特定频带的 EEG 信号转换为转换为 频谱图的形式。

![](EEGNetwork.png)

常见的，我们可以将 $\alpha$ 波段的信息转换为 频谱分析的形式，并借助卷积网络的方式将其编码为脉冲序列的形式。具体而言，```Fully Connection``` 的结果将作为 ```Spike Pattern``` 的结果。

频谱分析的方法可以通过如下的方式实现

```
## computer alpha bank (8-13hz)

window_data = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
window_data = np.abs(window_data) / args.window_length
window_data = np.sum(np.power(window_data[args.point_low:args.point_high, :], 2), axis=0)
```

### 基于端到端的EEG信号分类

对于端到端的EEG信号而言，考虑到 EEG 信号的时序性以及长序列特性，直接对原始信号进行直接编码需要更高的计算量。因而，我们可以借助 TAE 的三元脉冲编码的方式，将原本的长序列 EEG 信号转换为具有更高传输效率以及处理速度。我们可以借助 TAE.py 中的方法进行编码操作：

![](TAE.png)
```
from TAE import TAE

# tae: spike encoding， spare：spike fire ratio
tae, spare = TAE(train_set[0], alpha=1.5, thr=0.01)
```

## 搭建基于 Spike 神经元 的网络分类端

对于网络编码的结果，我们可以搭建基于脉冲神经元的网络分类架构。具体而言，网络的结构可以表示为：

```
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 16*16, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 2, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]
```
从而，我们可以借助上述的网络，得到对应的不同时间分辨率下的网络分类结果：
![](table.png)

