# LLM Quick Start

<p align="center">
    <br> English | <a href="README.md">中文</a>
</p>

Quick Start for Large Language Models (Theoretical Learning and Practical Fine-tuning)


## Setting Up the Development Environment

- Python v3.10+
- Python Environment Management: [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Interactive Python Development Environment: [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)
- [Audio processing toolkit ffmpeg](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)

For detailed installation instructions, please refer to [Documentation](docs/INSTALL.md)

### Installing Python Dependencies

Please use the `requirements.txt` file to install Python dependencies:

```shell
pip install -r requirements.txt
```

### About GPU Drivers and CUDA Versions

Typically, GPU drivers and CUDA versions need to meet the requirements of the installed PyTorch and TensorFlow versions.

Most recently released large language models use newer versions of PyTorch, such as PyTorch v2.0+. According to the PyTorch official documentation, the minimum required CUDA version is 11.8, along with a matching GPU driver version. You can find more details in the [PyTorch official CUDA version requirements](https://pytorch.org/get-started/pytorch-2.0/#faqs).

In summary, it is recommended to directly install the latest CUDA 12.3 version. You can find the installation packages on the [Nvidia official website](https://developer.nvidia.com/cuda-downloads).


After installation, use the `nvidia-smi` command to check the version:

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


### Configuring Jupyter Lab for Background Startup

After installing the development environment as mentioned above, it's recommended to start Jupyter Lab as a background service. Here's how to configure it (using the root user as an example):

```shell
# Generate a Jupyter Lab configuration file
$ jupyter lab --generate-config
Writing default config to: /root/.jupyter/jupyter_lab_config.py
```

Open the configuration file and make the following changes:

```python
# Allowing Jupyter Lab to start as a non-root user (no need to modify if starting as root)
c.ServerApp.allow_root = True
c.ServerApp.ip = '*'
```

Use `nohup` to start Jupyter Lab in the background:

```shell
$ nohup jupyter lab --port=8000 --NotebookApp.token='replace_with_your_password' --notebook-dir=./ &
```

Jupyter Lab's output log will be saved in the `nohup.out` file (which is already filtered in the `.gitignore` file).