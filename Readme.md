## Lingo: Make the LLM Better for Everyone🚀🚀

*Read this in [English](Readme_en.md).*

<div>



欢迎来到Lingo项目——让大型语言模型服务于人类的大舞台！🎉🎉

Lingo的核心使命是通过低成本的方式使得大型语言模型在更多领域发挥它的神奇力量。我们相信，只要稍微进行引导和微调，这些大规模的语言模型就能展现出令人惊艳的性能。💫🌈

在Lingo这个项目中，我们提供了超高质量的开源数据、高效微调代码以及微调后的模型权重。我们致力于为你提供最全面、最有效的工具和资源！🚀🚅



</div>

## 🔄 最近更新

* [2023/07/12] 开放Lingo训练代码，让我们轻松训练模型吧~
* [2023/06/20] 开放[Lingo-dataset-v1](https://huggingface.co/datasets/WENGSYX/Lingo-dataset-v1)，总计1091条高质量中文对话式问答训练集

### 如何安装

我们要求预先安装Apex和Deepspeed，安装教程可参考
```
git clone https://github.com/WENGSYX/Lingo
pip install .
```

### 零门槛训练大模型

Lingo是一个***零代码***+***零门槛***训练大模型的工具包。通过内置训练助手，你只需要用自然语言说明你的需求，Lingo就能全自动训练。
我们预先提供了一些IFT数据集，同时也可以使用自定义的数据加载

```bash
python -m lingo
```
或者使用python代码调用lingo：
```python
from lingo import let_lingo

let_lingo()
```


##### 然后就能够通过自然语言的形式，训练语言模型（在一台具有8卡A6000显卡的服务器中）
```
[INFO] This is a library for training language models with ease. 
[INFO] In conversations with Lingo, the language model will be trained automatically according to your needs, without requiring any effort on your part 😊
[INFO] Would you like to command Lingo through casual conversation? 
[Answer] If yes, please type (Yes), let"s go~, If not, please type (No): yes

[AI] Hello there! I"m your AI assistant, and I"m here to help you train your model. Before we get started, it"s important to have a clear plan and goal in mind. 
[Answer] : 我想要训练一个医疗领域的大模型

(AI] 好的，我们可以帮您选择医疗领域的大模型进行训练。首先，请问您打算使用我们提供的现有数据集 Ling-Hedical-V1，还是您有自己的数据集呢? 
[Answer] : 我想用自己的数据集

[AI] 好的，请将您的数据集配置为JSONL文件格式，然后将文件路径赋值给data字段，如果您不清楚如何配置，我可以为您解释。
[AI] JSON 是一种简单的文件格式，灵活且方便处理大型教据集。在您的教据集中，每一行应该是一个独立的字典，字典的键为"input和"output"，例:{'input': "What is hypertension","output": "Hypertension, also known as high blood pressure, is a medical condition in which the force of blood against the walls of the arteries is consistently elevated, leading to increased pressure. This can cause damage to the blood vessels and organs throughout the body over time."
[AI] 然后，您可以将数据集文件的绝对路径粘贴在这里，我会为您将它配置到模型训练中。
[Answer] : /data/LLM/Lingo/lingo.json

[AI] 已经将您的数据集路径设置为"/data/LLM/Lingo/lingojson"。接下来，我们需要为您的任务选择一个语言模型。模型的选择主要基于您的数据语言和显卡的显存。
[AI] 考虑到您想要训练医疗领域的大模型，我推荐使用以下模型: GLM-130B，Llama-13b，Llama-33b，或者Llama-65b。您的显卡的显存和数量如何?对于不同显存，建议的模型和训练方法有一些差别。
[AI] 您能方便告诉我你的显卡显存和数量吗? 这样我可以更准确地为您推荐模型和训练方法。
[Answer] : 我的服务器，现在有4张A6000显卡

[AI] 了解了，您的服务器上有4张A6000显卡，每张显卡的显存都是48GB。
[AI] 在这种配置下，您可以选择一些比较大的模型进行训练。例如，可以选择 130B 的模型，使用QLORA的方式来进行训练
[AI] 如果您对使用其他模型或训练方式有偏好，也可以告诉我，我们可以根据您的需要进行设定·您希望使用哪种模型进行训练呢?
[Answer] : 可以的

[AI] 已为您选择了GLH-130B模型进行训练。下一步，我们需要确定训练的方法。在当前的情况下，我建议使用LORA方法进行训练，QLORA方法仅微调模型的部分参数，可以建低显卡显存的要求，是最适合训练大植型的一种方法。如果您选择QLoRA训练方式，那我们推荐使用INT量化以优化显存。您是否同意使用QLoRA方法以及INT4量化？
[Answer] : 同意

...
(自动填写ds_config并调用Deepspeed使用QLoRA+GLM-130B训练模型)
```


### 创建你的特色数据集
我们还允许创建特色数据集，包括添加新的样本和指定模型名称等。
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

我们在[LIMA](https://huggingface.co/datasets/GAIR/lima)数据集的基础上人工翻译为中文问答，并在多处进行改写以适应中文环境，另外加入了一百条我们编写的高质量中文对话语料。

- 我们内置了数十条包含模型名字的样本，通过简单调用 `lingo_dataset.set_model_name`就可以一键为所有样本更新模型名字
- 我们支持额外添加新的样本，调用 `lingo_dataset.add_sample`并传入对话列表，即可自动加入新的对话样本。
- 一键获得数据集，调用 `lingo_dataset.get_list()`将返回列表格式的数据集，您可以在此基础上继续训练新的模型

### 🌱 Lingo's Roadmap 🌱

Version-1 目标 :

- [x] 开源高质量中文数据集
- [x] 开源模型的微调代码
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
