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
TORCH_CUDA_ARCH_LIST="7.5" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log


# 源代码安装 Transformers
# https://huggingface.co/docs/transformers/installation#install-from-source
pip install git+https://github.com/huggingface/transformers


# DeepSpeed ZeRO-2 模式单 GPU 训练翻译模型（T5-Small）
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro

# DeepSpeed ZeRO-2 模式单 GPU 训练翻译模型（T5-Large）
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero2.json \
--model_name_or_path t5-large \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--output_dir output_dir --overwrite_output_dir \
--do_train \
--do_eval \
--max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro \
--fp16 \



# DeepSpeed ZeRO-3 模式单 GPU 训练翻译模型（T5-Large）
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero3.json \
--model_name_or_path t5-3b --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro



# 直接使用 Python 命令启动 ZeRO-2 模式单 GPU 训练翻译模型（T5-Small）
python translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate