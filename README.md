# LLM-quickstart
大语言模型快速入门（理论学习与微调实战）


## 搭建开发环境

- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)


### 关于 GPU 驱动和 CUDA 版本

通常，GPU 驱动和 CUDA 版本都是需要满足安装的 PyTorch 和 TensorFlow 版本。

大多数新发布的大语言模型使用了较新的 PyTorch v2.0+ 版本，Pytorch 官方认为 CUDA 最低版本是 11.8 以及匹配的 GPU 驱动版本。详情见[Pytorch官方提供的 CUDA 最低版本要求回复](https://pytorch.org/get-started/pytorch-2.0/#faqs)。

简而言之，建议直接安装当前最新的 CUDA 12.2 版本，[详情见 Nvidia 官方安装包](https://developer.nvidia.com/cuda-downloads)。


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