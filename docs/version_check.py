import pkg_resources
import subprocess

# 首先，确保安装了 requirements.txt 中的所有包
subprocess.check_call(["pip", "install", "-r", "../requirements.txt"])

# 读取 requirements.txt 文件，获取软件包名称列表
with open("../requirements.txt", "r") as f:
    packages = f.readlines()
packages = [pkg.strip() for pkg in packages]

# 获取每个软件包的版本信息
with open("version_info.txt", "w") as output_file:
    for pkg in packages:
        if pkg == "" or pkg.startswith("#"):  # 跳过空行和注释
            continue
        try:
            # 尝试获取软件包版本
            version = pkg_resources.get_distribution(pkg).version
            output_file.write(f"{pkg}=={version}\n")
        except pkg_resources.DistributionNotFound:
            # 如果软件包未安装，则记录一个错误消息
            output_file.write(f"{pkg}: Not Found\n")

print("版本信息已写入 version_info.txt 文件。")
