################# 在编译和源代码安装 DeepSpeed 的机器运行 ######################3
# 更新 GCC 和 G++ 版本（如需）
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7 g++-7
# 更新系统的默认 gcc 和 g++ 指向
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc

# 源代码安装 DeepSpeed
# 根据你的 GPU 实际情况（查看方法见前一页），设置参数 TORCH_CUDA_ARCH_LIST；
# 如果你需要使用 NVMe Offload，设置参数  DS_BUILD_UTILS=1；
# 如果你需要使用 CPU Offload 优化器参数，设置参数 DS_BUILD_CPU_ADAM=1；
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.5" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 
python setup.py build_ext -j8 bdist_wheel
# 运行将生成类似于dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl的文件，
# 在其他节点安装：pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl。

# 源代码安装 Transformers
# https://huggingface.co/docs/transformers/installation#install-from-source
pip install git+https://github.com/huggingface/transformers


################# launch.slurm 脚本（按照实际情况修改模板值） ######################
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'