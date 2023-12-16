# LLM Quick Start
Quick Start for Large Language Models (Theoretical Learning and Practical Fine-tuning)

## Setting Up the Development Environment

- Python Environment Management: [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Interactive Python Development Environment: [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)

### About GPU Drivers and CUDA Versions

Typically, GPU drivers and CUDA versions need to meet the requirements of the installed PyTorch and TensorFlow versions.

Most recently released large language models use newer versions of PyTorch, such as PyTorch v2.0+. According to the PyTorch official documentation, the minimum required CUDA version is 11.8, along with a matching GPU driver version. You can find more details in the [PyTorch official CUDA version requirements](https://pytorch.org/get-started/pytorch-2.0/#faqs).

In summary, it's advisable to install the latest CUDA version, which is currently CUDA 12.2. You can find the installation packages on the [Nvidia official website](https://developer.nvidia.com/cuda-downloads).

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