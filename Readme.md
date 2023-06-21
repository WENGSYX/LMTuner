## Lingo: Make the LLM Better for Everyone🚀🚀

*Read this in [English](Readme_en.md).*

<div>



欢迎来到Lingo项目——让大型语言模型服务于人类的大舞台！🎉🎉

Lingo的核心使命是通过低成本的方式使得大型语言模型在更多领域发挥它的神奇力量。我们相信，只要稍微进行引导和微调，这些大规模的语言模型就能展现出令人惊艳的性能。💫🌈

在Lingo这个项目中，我们提供了超高质量的开源数据、高效微调代码以及微调后的模型权重。我们致力于为你提供最全面、最有效的工具和资源！🚀🚅



</div>

## 🔄 最近更新

* [2023/06/20] 开放[Lingo-dataset-v1](https://huggingface.co/datasets/WENGSYX/Lingo-dataset-v1)，总计1091条高质量中文对话式问答训练集

### 如何安装

```
git clone https://github.com/WENGSYX/lingo
pip install .
```

### 创建你的特色数据集

```python
from lingo.dataset import LingoDataset

lingo_dataset = LingoDataset()
# 给你的模型取一个名字
lingo_dataset.set_model_name('认知智能大模型')
# 增加问答数据集样本
lingo_dataset.add_sample(['你是谁？',
                          '大家好！我是一个超级棒的人工智能助手，认知智能大模型。我就像你的私人小助手，能用流利的自然语言和你聊天，无论是解答问题还是提供帮助，我都能轻松搞定。虽然我没有实体形象，但我会竭尽所能，为你提供最贴心的服务哦！'])

# 获得列表格式数据集
dataset = lingo_dataset.get_list()
```

我们在LIMA数据集的基础上人工翻译为中文问答，并在多处进行改写以适应中文环境，另外加入了一百条我们编写的高质量中文对话语料。

- 我们内置了数十条包含模型名字的样本，通过简单调用 `lingo_dataset.set_model_name`就可以一键为所有样本更新模型名字
- 我们支持额外添加新的样本，调用 `lingo_dataset.add_sample`并传入对话列表，即可自动加入新的对话样本。
- 一键获得数据集，调用 `lingo_dataset.get_list()`将返回列表格式的数据集，您可以在此基础上继续训练新的模型

### 🌱 Lingo's Roadmap 🌱

Version-1 目标 :

- [x] 开源高质量中文数据集
- [ ] 开源模型的微调代码
- [ ] 开源模型权重

Version-2 目标 :

- [ ] 数据集中加入Function Calling示例
- [ ] ...

### 引用

本项目为[神经理解](https://github.com/WENGSYX/Neural-Comprehension)的伴生项目。如果您对我们的项目感兴趣，欢迎引用。

```
@misc{weng2023mastering,
      title={Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks}, 
      author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},
      year={2023},
      eprint={2304.01665},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### 免责声明

**本项目相关资源仅供学术研究之用，严禁用于商业用途。**
使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。
