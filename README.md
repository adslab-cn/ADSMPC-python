# ADSMPC-python

## 环境配置
安装python环境（默认已经安装anaconda）
```bash
conda create --name ADSMPC-python python=3.10
conda activate ADSMPC-python
```

建议使用Pytorch==2.3.0
```bash
# CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch
```

导入NssMPC库
```bash
git clone https://github.com/XidianNSS/NssMPClib
```

安装必要的库
```bash
pip install -e .
```
如果报错，看看gcc是不是已经安装了
```bash
sudo apt-get install gcc
```

安装jupyter
调试代码时很好用的方式，可以一句一句执行代码
```bash
conda install ipython jupyter
```
