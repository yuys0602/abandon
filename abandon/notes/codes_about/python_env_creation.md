

## Anaconda 安装

1. Anaconda 用于 python 的环境管理，下载地址：[Anaconda](https://www.anaconda.com/products/individual)，[Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

1. linux 版本使用 `curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`，安装命令为 `bash Anaconda3-2019.03-Linux-x86_64.sh`。

1. 查看版本 `conda -V`

1. 更新 conda `conda update conda`

1. window 系统添加 conda 国内镜像，用于加速。

    1. 执行 `conda config --set show_channel_urls yes` 生成名为 `.condarc` 的文件。文件位于 `C:\Users\Administrator` 下。打开编辑内容，输入以下内容：

        ```
        # 清华源

        channels:
        - defaults
        show_channel_urls: true
        default_channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
        custom_channels:
        conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        ```
    
    1. 打开 `Anaconda Prompt` 窗口执行以下命令，注意不能使用 `Anaconda Powershell Prompt` 执行创建环境，会失败。

    1. 运行 `conda config --show-sources` 查看 config 配置是否正确。

    1. 运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引。

    1. 运行 `conda env list` 查看现有环境列表，用来保证下一步测试环境名不重复。

    1. 运行 `conda create -n testenv numpy` 测试一下速度。

    1. 执行 `conda activate testenv` 运行环境

    1. 执行 `conda deactivate` 退出测试环境

    1. 执行 `conda remove -n testenv --all` 删除测试环境

# Cuda 和 Cudnn 安装

1. 添加环境变量。右键 `我的电脑` ，点击`高级系统设置`下的`高级`的`环境变量`。

    在`用户变量`中添加如下环境：

    ```
    CUDA_PATH
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1

    NVCUDASAMPLES_ROOT
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.1
    ```

    在`系统变量`中 `Path`编辑添加如下环境：

    ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64

    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include

    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\CUPTI\lib64

    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.1\bin\win64

    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.1\common\lib\x64
    ```

1. 查看 cuda 版本 `cuda -V`。

1. 查看 cudnn 版本 打开`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include\cudnn_version.h`。


## Pip 加速

window 系统在 `C:\Users\Administrator\pip\pip.ini` 输入

```
[global]
 
index-url = https://mirrors.aliyun.com/pypi/simple/
 
[install]
 
trusted-host=mirrors.aliyun.com
```

## Pytorch 安装

> 安装的是 `cuda11.1` 和 `cudnn8.0.5`， 时间 `2021/03/21`。

1. 首先使用 Anaconda 创建新的环境，在此之前可以先查看现有环境列表，删除自己不想要的环境。

1. 执行 `conda create -n py39torch18 python==3.9.2` 创建 `py3.9.2` 环境。

1. 激活环境 `conda activate py39torch18`。

1. 安装 pytorch，执行 `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`。比较慢，一共2.95G。

1. 可以将当前环境备份 `>conda create -n py39torch18archive --clone py39torch18`。

1. window 系统使用 `nvidia-smi`，需要现在环境变量中添加 `C:\Program Files\NVIDIA Corporation\NVSMI`， 之后使用 `nvidia-smi.exe` 命令查看。

1. 测试 gpu 环境是否安装正确。

    ```python
    import torch
    
    print(torch.__version__) # 1.8.0
 
    print(torch.version.cuda) # 11.1
 
    print(torch.cuda.is_available()) # True
    ```

# Pycharm 配置

1. 头文件设置，进入File -->settings-->Editor-->File and Code Templates-->Python Script

```
# coding=utf-8
# Copyright (C) xxx team - All Rights Reserved
#
# @Version:   3.9.2
# @Software:  ${PRODUCT_NAME}
# @FileName:  ${NAME}.py
# @CTime:     ${DATE} ${TIME}   
# @Author:    xxx
# @Email:     xxx
# @UTime:     ${DATE} ${TIME}
#
# @Description:
#     xxx
#     xxx
#
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def main():
    pass



if __name__ == '__main__':
    main()
```


# 构建环境

## conda 环境

```
conda env list

conda create -n py10pt11 python=3.10
# 默认安装python3.10.4，是3.10中最新的版本

pip install -U torch torchvision torchaudio pytorch-lightning transformers hydra_core matplotlib scikit-learn onnx onnxruntime ipdb
```
