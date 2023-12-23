# 大模型微调训练 快速入门

<p align="center">
    <br> 中文 | <a href="README-en.md">English</a>
</p>


大语言模型快速入门（理论学习与微调实战）

## 搭建开发环境

- Python v3.10+
- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)
- [音频处理工具包 ffmpeg](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)

详细安装说明请参考[文档](docs/INSTALL.md)

### 安装 Python 依赖包

请使用 `requirements.txt` 文件进行 Python 依赖包安装：

```shell
pip install -r requirements.txt
```

### 关于 GPU 驱动和 CUDA 版本

通常，GPU 驱动和 CUDA 版本都是需要满足安装的 PyTorch 和 TensorFlow 版本。

大多数新发布的大语言模型使用了较新的 PyTorch v2.0+ 版本，Pytorch 官方认为 CUDA 最低版本是 11.8 以及匹配的 GPU 驱动版本。详情见[Pytorch官方提供的 CUDA 最低版本要求回复](https://pytorch.org/get-started/pytorch-2.0/#faqs)。

简而言之，建议直接安装当前最新的 CUDA 12.3 版本，[详情见 Nvidia 官方安装包](https://developer.nvidia.com/cuda-downloads)。

安装完成后，使用 `nvidia-smi` 指令查看版本：

```shell
nvidia-smi          
Mon Dec 18 12:10:47 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:0D.0 Off |                    0 |
| N/A   44C    P0              26W /  70W |      2MiB / 15360MiB |      6%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

### Jupyter Lab 后台启动配置

上述开发环境安装完成后，建议使用后台常驻的方式来启动 Jupyter Lab。下面是相关配置（以 root 用户为例）：

```shell
# 生成 Jupyter Lab 配置文件，
$ jupyter lab --generate-config
Writing default config to: /root/.jupyter/jupyter_lab_config.py
```

打开配置文件后，修改以下配置项：

```python
# 非 root 用户启动，无需修改
c.ServerApp.allow_root = True
c.ServerApp.ip = '*'
```

使用 nohup 后台启动 Jupyter Lab
```shell
$ nohup jupyter lab --port=8000 --NotebookApp.token='替换为你的密码' --notebook-dir=./ &
```

Jupyter Lab 输出的日志将会保存在 `nohup.out` 文件（已在 .gitignore中过滤）。