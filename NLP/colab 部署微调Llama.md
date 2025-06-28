

### Unsloth教程代码

目前，最简单的方法是使用Unsloth，它是一个微调模型的集成工具。通过Unsloth微调Mistral、Gemma、Llama，速度提高2-5倍，内存减少70%。

Unsloth的github上有适合新手的Colab训练脚本：https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing，照着一步步执行就可以顺利微调成功

简化后的代码如下：

#### 安装unsloth

```python
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# We have to check which Torch version for Xformers (2.3 -> 0.0.27)
from torch import __version__; from packaging.version import Version as V
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
!pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton
```

#### 下载预训练模型

默认已选择unsloth/Meta-Llama-3.1-8B模型

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

#### 设置LoRA训练参数

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

#### 数据准备

使用来自[yahma](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fyahma%2Falpaca-cleaned)的Alpaca数据集，这是原始[Alpaca数据集](https://www.google.com/url?q=https%3A%2F%2Fcrfm.stanford.edu%2F2023%2F03%2F13%2Falpaca.html)的经过筛选的版本，包含了来自原始数据集中的52000条数据。可以用自己的数据准备替换此代码部分。

**[注意]** 要仅训练完成（忽略用户输入）

**[注意]** 记得将**EOS_TOKEN**添加到标记化的输出中！否则将得到无限的生成！

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

#### 训练模型

使用Huggingface TRL的`SFTTrainer`。执行60步来加快速度，但可以设置`num_train_epochs=1`进行完整运行，并关闭`max_steps=None`。还支持TRL的`DPOTrainer`

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

#### 开始训练

```python
trainer_stats = trainer.train()
```

#### 测试训练效果

运行模型，可以更改指令（instruction）和输入（input） - 将输出（output）留空

```python
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

还可以使用`TextStreamer`进行连续推断 - 这样可以逐个标记地查看生成过程，而不必等待全部生成完成

```python
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

#### 保存、加载新模型

要将最终模型保存为 LoRA 适配器，使用 `save_pretrained` 进行本地保存。

检测当前地址并保存模型：

```python
import os
#获取当前的保存的地址
current_path = os.getcwd()
model.save_pretrained("lora_model") # 本地保存
print(f"保存地址 {current_path}/lora_model")

```

#### 测试LoRA模型

```python
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # 你的训练的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # 指令 instruction
        "", # 输入 input
        "", # 输出 output - 留空!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
```

#### 移动合成的模型到谷歌云硬盘，便于下载到本地：

```python
# Save to q4_k_m GGUF 推荐（完成后将模型放到谷歌云硬盘）
if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

# Save to 16bit GGUF（这个体积太大了）
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
```

#### 16模型量化成Q8，减少模型体积（如果是Q4可以不用）

训练完成model-unsloth.F16.gguf太大了，免费的谷歌云无法保存，量化成Q8可以减少一半的大小

```python
![ -d "llama.cpp" ] || git clone https://github.com/ggerganov/llama.cpp.git
!cd llama.cpp && make
#!./quantize /content/model-unsloth.F16.gguf /content/model-unsloth.q8_0.gguf q8_0
```

#### 将刚刚生成的模型移动到谷歌云

source_path 是你的模型地址 destination_path 是谷歌云地址

```python
import shutil

source_path = '/content/model-unsloth.Q4_K_M.gguf'
destination_path = '/content/drive/MyDrive/'

# 移动文件，内容有点大需要点时间
shutil.move(source_path, destination_path)
print("请使用谷歌云MyDrive中下载该内容")
```

## 使用本地数据集微调：

基本步骤同上，主要是数据集的不同

```python
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
 
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
 
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth
 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
 
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
 
 
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
 
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        input = ""
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
 
from datasets import load_dataset
#dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
#dataset = dataset.map(formatting_prompts_func, batched = True,)
from google.colab import drive
# 挂载云端硬盘，加载成功后，在左边的文件树中将会多一个 /content/drive/MyDrive/ 目录
drive.mount('/content/drive')
 
 
# 加载本地数据集：
# 有instruction和output,input为空字符串
from datasets import load_dataset
 
data_home = r"/content/drive/MyDrive/"
data_dict = {
    "train": os.path.join(data_home, "train.json"),
    #"validation": os.path.join(data_home, "dev.json"),
}
dataset = load_dataset("json", data_files=data_dict, split = "train")
print(dataset[0])
dataset = dataset.map(formatting_prompts_func, batched = True,)
 
 
from trl import SFTTrainer
from transformers import TrainingArguments
 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
 
# 开始微调训练
trainer_stats = trainer.train()
 
#推理
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "你是谁？", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")
 
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
 
#此处输出的答案，能明显看到就是自己训练的数据，而不是原来模型的输出。说明微调起作用了
 
 
# 保存模型，改成挂接的云硬盘目录也可以保存到google的个人云存储空间，然后打开个人云存储空间下载到本地
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
 
# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
```



