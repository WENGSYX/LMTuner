# 如何在服务器中从零安装Lingo所需环境 (Deepspeed 和 Apex 安装教程)

***在本文档中，我们将详细介绍如何为 Lingo 框架安装基础环境。我们将介绍每个阶段的步骤和可能遇到的问题。***



### 环境安装

##### 安装GPU驱动

```bash
# preparing environment
sudo apt-get install gcc
sudo apt-get install make
wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
sudo sh cuda_11.5.1_495.29.05_linux.run
```
如果此步无sudo权限，可使用源码安装的方式。请注意设置安装路径细节等。并在安装完毕后将安装路径加入至PATH路径中。
##### 安装conda和python

```bash
# preparing environment
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n Lingo python==3.9
conda activate Lingo
```

##### 安装python库

```bash
# preparing environment
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install tqdm transformers sklearn pandas numpy accelerate sentencepiece wandb SwissArmyTransformer jieba rouge_chinese datasets
```

如果确保安装正确的GCC库（>5.0.0）后，可继续安装apex和deepspeed
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install deepspeed
```
请先安装apex，再安装deepspeed

在安装apex时：

1. 如果提醒"Pytorch binaries were compiled with Cuda“表示Pytorch版本可能跟cuda不符，但此问题不会对apex的安装造成影响，因此将apex的setup.py文件第32行”    if (bare_metal_version != torch_binary_version):“替换为”    if 0:“

2. No module named 'packaging'：pip install packaging

3. ninja: build stopped: subcommand failed.: pip install ninja

4. subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.: 修改torch内的cpp文件,变为ninja -V

5. g++: error: /home/wengyixuan/apex/build/temp.linux-x86_64-cpython-39/csrc/fused_dense.o: No such file or directory: 放弃吧，八字不合，要么torch环境不对,要么pytorch环境不对，要么apex环境不对。建议换成其他版本，如python=3.8，torch==1.7/1.8或者apex的历史版本 (这是因为未来可能更新新的版本)

##### 安装Lingo库
```bash
git clone https://github.com/WENGSYX/Lingo
cd LMTuner
pip install .
```

### 多机多卡时 hostfile内容


hostfile（以两台八卡服务器为例）：
```
host1 slots=8 #表示使用host1中的8张显卡
host2 slots=8 #表示使用host2中的8张显卡
```
请注意，deepspeed使用ssh控制其他服务器，因此使用多机时需要确保主服务器能够通过无密码ssh连接其他服务器。


### 开始训练 
- 在训练时，请密切关注 GPU 使用率、内存使用情况和网络通信状况。确保利用率保持在较高水平。
- 关注训练过程中的性能指标，如PPL（困惑度）、损失等。定期检查训练是否按预期进行。
- 储存检查点和日志，以便在训练失败或需要重启时，可以从中断处继续训练。
- 要确保代码质量和良好的训练效果，定期验证和测试代码。


### 7. 可能遇到的问题和解决方案
- OOM（内存溢出）：减小批量大小、使用梯度累计或调整模型参数。
- 梯度爆炸/消失：使用梯度裁剪，适当调整学习率。
- 过拟合：考虑增加正则化手段（如 L2 正则化、Dropout 等），使用更大的数据集。
- 低 GPU 利用率：检查数据加载和批处理大小，避免小批量训练。
- 混合精度训练问题：确保已启用混合精度选项，检查数值稳定性，使用梯度缩放。
- 网络通信瓶颈：优化数据加载过程，使用专用的通信库（如 NCCL）和高速互联。
- 剩余问题，建议随时询问GPT-4

#### pdsh安装	
RuntimeError: launcher 'pdsh' not installed.
pdsh安装：https://www.dandelioncloud.cn/article/details/1538030758489079809

安装：
./configure –-with-ssh –-enable-static-modules –-prefix=/home/username && make && make install
Because we have installed in home dir, need to add /home/username/bin to system path,
add the following to your .bashrc: export PATH=$PATH:/home/username/bin
软连接sudo ln -s /home/username/bin/pdsh /usr/bin/
测试：pdsh -V

#### 为什么deepspeed的zero2不支持pipeline

zero2需要做梯度的all_reduce，如果做了流水线并行，会造成频繁的梯度all_reduce

操作，导致并不能取得一个很好的平衡，所以一般是mp+pp+zero1

#### nccl环境设置
	https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

#### torch.distributed包
https://pytorch.org/docs/stable/distributed.html#

https://www.jianshu.com/p/5f6cd6b50140

#### 无法使用单机多卡
可能由于p2p传输，设置NCCL_P2P_DISABLE = 1


#### 训练时学习率为0
可能是梯度消失，建议检查模型、优化器等，并切记使用高斯初始化权重（如果新增权重）

