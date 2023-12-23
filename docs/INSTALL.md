# Transformers开发环境搭建
## 介绍
开发环境搭建包含几个部分
- Miniconda
- Jupyter Lab
- Hugging Face Transformers，需要尝试多种模型时候，建议tensorflow和pytorch都安装
- 其他依赖包

## Miniconda
Miniconda 是一个 Python 环境管理工具，可以用来创建、管理多个 Python 环境。它是 Anaconda 的轻量级替代品，不包含任何 IDE 工具。 Miniconda可以从[官网](https://docs.conda.io/en/latest/miniconda.html)下载安装包。也可以从镜像网站下载：

### Miniconda环境的安装
```bash
# 下载 Miniconda 安装包
$ wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 也可以使用curl命令下载
$ curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 安装 Miniconda
$ bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中，需要回答一些问题，如安装路径、是否将 Miniconda 添加到环境变量等。安装完成后，需要重启终端，使环境变量生效。

可以使用以下命令来验证 Miniconda 是否安装成功：

```bash
$ conda --version
```

### 配置Miniconda
Miniconda的配置文件存放在~/.condarc，可以参考文档手工修改，也可以使用conda config命令来修改。

1. 为了加速包下载，可以配置使用国内的镜像源：
```bash
# 配置清华镜像
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
$ conda config --set show_channel_urls yes
# 查看~/.condarc配置
$ conda config --show-sources
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: True
```
2. 加速anaconda包的下载
可以使用mamba或micromamba来代替conda，这两个工具都是conda的替代品，会缓存包的版本信息，不需要在每次安装包的时候都去检查，这种可以有效提高conda-forge这种比较大的。安装mamba或micromamba的方法如下：
```bash
# 安装mamba
$ conda install -n base -c conda-forge mamba
# 安装micromamba
$ conda install -n base -c conda-forge micromamba
```
之后可以使用mamba或者micromamba命令代替conda命令。

### 创建虚拟环境
```bash
# 创建虚拟环境，指定 Python 版本为 3.11
(base) $ conda create -n transformers python=3.11
# 激活 openai 环境
$ conda activate transformers
```
以下若无特殊说明，均在这里新建的openai环境中进行。

## Jupyter Lab
Jupyter Lab 是一个交互式的开发环境，可以在浏览器中运行。它支持多种编程语言，包括 Python、R、Julia 等。 Jupyter Lab由conda-forge提供，请先配置镜像，然后使用以下命令安装：
```bash
(transformers) $ conda install jupyterlab
```

## Hugging Face Transformers
Hugging Face Transformers 是一个基于 PyTorch 和 TensorFlow 的自然语言处理工具包，提供了大量预训练模型，可以用来完成多种 NLP 任务。Hugging Face Transformers 可以通过 conda 安装：

```bash
(transformers) $ conda install -c huggingface transformers
```

安装文档：[Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)

## 安装tensorflow
Transformers需要使用tensorflow进行实际的模型推理，以下命令安装tensorflow的CPU和GPU版本：
```bash
(transformers) $ pip install tensorflow
```

若是使用Mac，对M1/M2芯片可以安装Metal插件，一些小一些的模型也可以尝试：
```bash
(transformers) $ pip install tensorflow-metal
```
安装文档：
- [tensorflow](https://www.tensorflow.org/install)
- [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
## 安装pytorch
Transformers需要使用pytorch进行实际的模型推理，在前面已经配置了使用的pytorch和conda-forge镜像源，可以使用下命令安装和CUDA版本对应的Pytorch版本：
```bash
# Linux
# CUDA 11.8
(transformers) $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c nvidia
# CUDA 12.1
(transformers) $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia

# Mac
(transformers) $ conda install pytorch::pytorch torchvision torchaudio
```

安装文档：[pytorch](https://pytorch.org/get-started/locally/)

## 安装其他的依赖包
在处理图像、音频等数据时，需要使用到其他的依赖包，包括：
- tqdm、iprogress 进度条
- ffmpeg、ffmpeg-python 音频处理工具
- pillow 图像处理工具

```bash
(transformers) $ conda install tqdm iprogress ffmpeg ffmpeg-python pillow
```
