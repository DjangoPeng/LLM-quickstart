# LLM-quickstart
大语言模型快速入门（理论学习与微调实战）



## 搭建开发环境

- Python v3.10+
- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)
- [音频处理工具包 ffmpeg](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)

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