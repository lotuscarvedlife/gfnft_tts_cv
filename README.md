# 项目说明文件

## 一、环境配置

- CosyVoice 的环境较为简单，请参考语雀文档

  ```shell
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
  cd CosyVoice
  git submodule update --init --recursive
  conda create -n cosyvoice -y python=3.10
  conda activate cosyvoice
  conda install -y -c conda-forge pynini==2.1.5
  pip install -r requirements.txt
  
  mkdir -p pretrained_models
  git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
  ```

- Next_sentence 的环境还是按照之前的那个，只不过要配合新的 torch

  ```shell
  pip install lightning torch==2.3.1
  pip install torchdata torch==2.3.1
  conda install editdistance -c conda-forge
  conda install wandb -c conda-forge
  pip install sentence-transformers datasets loralib accelerate matplotlib nltk peft spacy   
  # 最后一个 requirements.txt 好像没写，但是确实需要
  ```

- 有几个补充的：

  ```python
  sys.path.append('third_party/Matcha-TTS') # train.py 文件需要加上这个
  ```



## 二、代码修改

> 本部分集中说明我们在尝试各种方法，需要改动代码时可以改动哪些，有部分代码在最初适配时完成修改后一直暂时没有修改的，在这里将不进行说明，如果之后有需要将会进行补充

### 2.1 训练部分

- configs
  - device
    - `gpu.yaml`：配置训练使用的 gpu 数量（在运行推理的时候需要使用 CUDA_VISIBLE_DEVICES 指定 gpu id）
  - task
    - `gfnft_cosyvoice.yaml`：训练的主要配置文件，其中包括了：lora 微调层、一次解码（采样）数量、梯度累计步数、训练总 epoch 数、reward 温度变化范围和变化速度、模型 checkpoint 记录、最大采样长度、非法 token（用于设置静音 penalty） 等。
- data_preprocess
  - train-clean-100-libritts（这一部分的代码请在当前文件夹下，配合所有数据集运行）
    - `_***`：这类文件用于统计数据集信息
    - `num_***`：这类文件用于从 LibriTTS 数据集中指定长度等信息挑选自己需要的数据集，请按照最开始的序号依次运行。
- `data_preprocessor.py`：数据的预处理代码，用于制作模型训练的 tensor 并保存，这样在训练的过程中就不需要调用 tokenizer 等多余的模型文件，并加速模型的训练了。需要配合数据集使用。可以修改增加传入的数据内容
- `train.py`：训练的入口文件，可以指定是否使用非法 token mask（用于设置静音等 token 的 penalty）、训练前的 sanity val check 验证步数等
- `lightning_module.py`：训练的主要框架，这里我们可以设置使用的loss，调整学习率变化，调整 reward 温度变化
- `utils.py`：训练的工具，大部分重要代码都在这里，如调整 reward、调整最小长度的 reward、调整训练时候的最小长度、调整计算loss中间的过程等。

### 2.2 推理部分

- cosyvoice
  - cli
    - `model.py`：调用推理函数的文件，可以调整是否加载 lora 模型（load 函数中直接进行注释）、是否使用简单采样、计算 reward 时是否使用 base 模型、是否使用特殊 token penalty，对哪些 token 施加 penalty（注意，这里 penalty 同时包括采样概率和 reward）
    - `cosyvoice.py`：我们可以在这里指定加载什么 lora 模型（指定模型名称，这里如果设置为 None 的话也可以让推理不再加载 lora 模型
  - llm
    - `llm.py`：推理函数就在这里，我们可以调整推理的时候，推理要求的最小句子长度，打印在推理的时候的 reward、概率等信息
- `test.py`：测试文件，在这里可以调整推理的目标、调整推理出的结果的存储位置等。