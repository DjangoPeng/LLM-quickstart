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

目前支持项目运行的软件版本列表如下所示，详见[版本对照文档](docs/version_info.txt)：

```makefile
torch>=2.1.2==2.3.0.dev20240116+cu121
transformers==4.37.2
ffmpeg==1.4
ffmpeg-python==0.2.0
timm==0.9.12
datasets==2.16.1
evaluate==0.4.1
scikit-learn==1.3.2
pandas==2.1.1
peft==0.7.2.dev0
accelerate==0.26.1
autoawq==0.2.2
optimum==1.17.0.dev0
auto-gptq==0.6.0
bitsandbytes>0.39.0==0.41.3.post2
jiwer==3.0.3
soundfile>=0.12.1==0.12.1
librosa==0.10.1
langchain==0.1.0
gradio==4.13.0
```

为了检查你的运行环境中软件版本是否匹配，项目提供了自动化[版本检查脚本](docs/version_check.py)，请注意修改输出文件名。


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

### 关于 LangChain 调用 OpenAI GPT API 的配置

为了使用OpenAI API，你需要从OpenAI控制台获取一个API密钥。一旦你有了密钥，你可以将其设置为环境变量：

对于基于Unix的系统（如Ubuntu或MacOS），你可以在终端中运行以下命令：

```bash
export OPENAI_API_KEY='你的-api-key'
```

对于Windows，你可以在命令提示符中使用以下命令：

```bash
set OPENAI_API_KEY=你的-api-key
```

请确保将`'你的-api-key'`替换为你的实际OpenAI API密钥。