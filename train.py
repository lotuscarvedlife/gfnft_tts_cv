from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
import os
import sys
import logging
import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from cosyvoice.utils.file_utils import load_wav
from utils import (
    FrozenModelSentenceGivenPrompt,
    RuleSentenceValidator,
    ModelSentenceValidator,
    ReplayBuffer,
)
from lightning_module import NextSentenceGFNTask
from lightning_data import PromptDataModule
from hyperpyyaml import load_hyperpyyaml
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.frontend import CosyVoiceFrontEnd

# 让我看看你到底是什么东西
def print_all_and_exit(stop: bool, *args, **kwargs):
    if args:
        print("位置参数:")
        for i, arg in enumerate(args, start=1):
            if isinstance(arg, list):
                print(f"  参数 {i}: {arg} (类型: {type(arg)}, 长度：{len(arg)})")
            elif isinstance(arg, torch.Tensor):
                print(f"  参数 {i}: {arg} (类型: {type(arg)}, 形状：{arg.shape})")
            else:
                print(f"  参数 {i}: {arg} (类型: {type(arg)})")
    if kwargs:
        print("关键字参数:")
        for key, value in kwargs.items():
            if isinstance(value, list):
                print(f"  {key}: {value} (类型: {type(value)}, 长度：{len(value)})")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value} (类型: {type(value)}, 形状：{value.shape})")
            else:
                print(f"  {key}: {value} (类型: {type(value)})")
    if stop:
        import sys
        sys.exit("调用打印后退出，进程正常停止。")


def wrap_cuda_model(train_engine, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
    return model


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def train(config: DictConfig):
    # 设置全局随机种子
    pl.seed_everything(config.seed, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tts_text = "I do not know whether you would mind that."
    # prompt_text = "A complete change of life became desirable."
    # prompt_speech_16k = load_wav('data/audio/39_121915_000005_000000.wav', 16000)
    # sample_rate= 24000
    # min_token_text_ratio = 2
    # max_token_text_ratio = 20
    
    # 加载模型和tokenizer
    model = get_model(config)
    model.to(device)

    # model_dir = config.task.model.model_dir
    # with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
    #     configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    
    # frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
    #                                       configs['feat_extractor'],
    #                                       '{}/campplus.onnx'.format(model_dir),
    #                                       '{}/speech_tokenizer_v2.onnx'.format(model_dir),
    #                                       '{}/spk2info.pt'.format(model_dir),
    #                                       configs['allowed_special'])    

    # prompt_text = frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    # tts_text = frontend.text_normalize(tts_text, split=False, text_frontend=True)

    # # 输入准备
    # model_input = frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, sample_rate)
    # text = model_input["text"].to(device)
    # text_len = torch.tensor([text.shape[1]], dtype=torch.int32).to(text.device)
    # prompt_text = model_input["prompt_text"].to(device)
    # prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(prompt_text.device)
    # llm_prompt_speech_token = model_input["llm_prompt_speech_token"].to(device)
    # prompt_speech_token = llm_prompt_speech_token
    # prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(prompt_speech_token.device)
    # llm_embedding = model_input["llm_embedding"].to(device)
    # embedding = llm_embedding

    # # 输入衔接与处理(llm.inference)
    # text = torch.concat([prompt_text, text], dim=1)
    # text_len += prompt_text_len
    # print(text.device)
    # text = model.llm.model.model.embed_tokens(text)
    # embedding = torch.zeros(1, 0, model.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
    # sos_eos_emb = model.llm_embedding.weight[model.sos_eos].reshape(1, 1, -1)
    # task_id_emb = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)
    # if prompt_speech_token_len != 0:
    #     prompt_speech_token_emb = model.speech_embedding(prompt_speech_token)
    # else:
    #     prompt_speech_token_emb = torch.zeros(1, 0, model.llm_input_size, dtype=text.dtype).to(device)
    # lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
    
    
    
    
    # lm_input = torch.load("data/lm_input_tensor/data_39_121915_000005_000000.pt")




    # min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
    # max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

    # cache = None
    # # 模型输入
    # out_tokens = []
    # y_pred, cache = model.llm.forward_one_step(lm_input,    # torch.Size([1, 149, 896])
    #                                             masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
    #                                             cache=cache)
    # logp = model.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
    # top_ids = model.sampling_ids(logp.squeeze(dim=0), out_tokens, 25, ignore_eos=True if 0 < min_len else False)

    # # lm_input = model.speech_embedding.weight[top_ids].reshape(1, 1, -1)
    
    # print_all_and_exit(True, lm_input=lm_input, y_pred=y_pred, logp=logp, top_ids=top_ids)
    # # print_all_and_exit(True, prompt_text=prompt_text, tts_text=tts_text, model_input=model_input, model_output=model_output)





    # 结束符
    end_of_sentence_token_id = model.speech_token_size


    
    # # 创建一个布尔数组表示哪些Token被认为是非法的，并将其转换为NumPy数组以便后续使用。
    # # vocab_size是tokenizer的词汇表大小
    # illegal_token_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    # # 获取设置里提供的非法的Token，并转为 python 容器。
    # illegal_tokens = OmegaConf.to_container(config.task.constraints.illegal_tokens)
    # # 遍历非法 token 列表，如果是整数则直接添加 [t]，否则（应该是字符串形式的）将其编码为Token ID。
    # illegal_tokens = [
    #     [t] if isinstance(t, int) else tokenizer.encode(t, add_special_tokens=False)
    #     for t in illegal_tokens
    # ]
    # # 验证每个子列表长度都为1
    # assert all(len(t) == 1 for t in illegal_tokens)
    # # 获取非法 token 的 ID，直接组成一个列表
    # illegal_tokens = [t[0] for t in illegal_tokens]
    # # 将掩码中非法 token 的 ID 对应的位置标记为True
    # illegal_token_mask[illegal_tokens] = True
    # # 将掩码转换为NumPy数组
    # illegal_token_mask = illegal_token_mask.numpy()

    illegal_token_mask = None

    # 设置奖励函数
    reward = get_reward(config, end_of_sentence_token_id, illegal_token_mask)
    # 创建一个奖励缓存，用于存储 token 和对应的奖励。
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,         # 缓存大小，设定为50
        termination_token_id=end_of_sentence_token_id,      # 句子结束 token 的 ID
    )

    # 根据配置加载数据集并划分训练集和验证集。
    data = PromptDataModule(
        data_path=config.task.data.path,                # 数据存储路径，存储的是 prompt
        train_size=config.task.data.train_size,         # 训练集大小比例，设定为 0.95
        # val_size=config.task.data.val_size,             
        limit_prompts=config.task.data.limit_prompts,   # 限制 prompt 数据集大小
    )
    # 进行 fit 阶段，即开始准备训练集和验证集，不过原本的函数中没有使用就是了
    data.setup("fit")
    
    # NOTE: 探针没有适配
    # 训练集和测试集探针，为列表，提取前 config.task.eval.n_probes 个 prompt 编码 token_id 列表（[0] 是消除批次维度）
    # train_probes = [data.train_data[i][0] for i in range(config.task.eval.n_probes)]
    # val_probes = [data.val_data[i][0] for i in range(config.task.eval.n_probes)]
    train_probes = None
    val_probes = None

    # 创建一个训练任务实例，包含了模型、分词器、奖励机制等所有需要的信息。
    task = NextSentenceGFNTask(
        model=model,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        illegal_token_mask=illegal_token_mask,
        train_probes=train_probes,
        val_probes=val_probes,
        diversity_metric=None,
        use_4bit=config.task.training.use_4bit,
        devices_count=config.device.count
    )

    # 创建一个PyTorch Lightning Trainer实例，配置了加速器、最大轮数、梯度累积步数、日志记录器和回调函数等。
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,                                          # 运行命令指定gpu.yaml，加速器为 cuda
        max_epochs=config.task.training.epochs,                                         # config 文件中指定为 300
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,           # 运行命令有指定为 32，指定梯度累计的步数
        num_sanity_val_steps=0,
        logger=config.logger                                                            # 在 train.yaml 中指定了 wandb
        if isinstance(config.logger, bool)                                              # 意思是如果 config.logger 是布尔值，则不加载 logger
        else hydra.utils.instantiate(config.logger),                                    # 否则就加载指定的 logger
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],          # 指定训练过程中使用的回调函数，包括保存模型权重，监控指标提早停止训练等函数
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    trainer.fit(model=task, datamodule=data)


# 根据配置加载预训练模型和分词器，并应用4位量化和LoRA技术。
def get_model(config: DictConfig):
    # Use 4-bit quantization for lower memory use，在本次训练中没有使用这个技术，技术是用来降低内存使用的
    if config.task.training.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None


    # # Get the model
    # # 加载分词器，名字为 gpt2-xl，默认不添加 beginning of sentence token (BOS)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     config.task.model.name, add_bos_token=False
    # )

    # override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}
    model_dir = config.task.model.model_dir
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    
    state_dict = torch.load('{}/llm.pt'.format(model_dir), map_location='cuda')
    model = configs["llm"]
    model.load_state_dict(state_dict, strict=False)
    

    # # Wrap using Lora
    # # LoRA: Low Rank Adaptation of Large Language Models; peft: parameter efficient fine-tuning
    # # 一种用于微调预训练模型的技术，通过引入低秩矩阵，以减少模型的参数数量，从而降低模型的训练时间和推理时间。
    model = get_peft_model(
        model, hydra.utils.instantiate(config.task.model.lora_config)
    )

    # # Remove dropout
    # # 移除模型中所有 dropout 层，但好像也把 lora 的 dropout 层也移除了？
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0


    # model = wrap_cuda_model("torch_ddp", model)


    # 加载预训练模型，名字为 gpt2-xl，自动分配设备（cpu或gpu），配置4位量化
    # model = AutoModelForCausalLM.from_pretrained(
    #     config.task.model.name, device_map="auto", quantization_config=bnb_config
    # )

    # Prepare model for k-bit training
    # 准备模型以进行k-bit训练。
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )


    return model

# 根据配置选择合适的句子验证器并初始化奖励机制。
def get_reward(config: DictConfig, sentence_token_id, illegal_token_mask):
    # 根据配置选择句子验证器 sentence validator，本次配置为 null
    if config.task.reward.sentence_validator is None:
        sentence_validator, valid_sentence_alpha = None, None
    elif config.task.reward.sentence_validator == "rule":
        sentence_validator, valid_sentence_alpha = (
            RuleSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    elif config.task.reward.sentence_validator == "model":
        sentence_validator, valid_sentence_alpha = (
            ModelSentenceValidator(sentence_token_id=sentence_token_id),
            config.task.reward.valid_sentence_alpha,
        )
    else:
        raise ValueError(
            f"Invalid sentence validator: {config.task.reward.sentence_validator}"
        )

    reward = FrozenModelSentenceGivenPrompt(
        temperature=1.125,
        sentence_token_id=sentence_token_id,                    # 输入语句结束 token id（“.”）
        min_len=config.task.constraints.min_sentence_len,       # 最小句子长度，为1
        vocab_alpha=config.task.reward.vocab_alpha,             # 词汇表偏好的概率补偿
        vocab_naughty_mask=illegal_token_mask,                  # 非法词汇表掩码
        sentence_validator=sentence_validator,                  # 句子验证器
        valid_sentence_alpha=valid_sentence_alpha,              # 验证句子的有效性权重参数
    )

    return reward


if __name__ == "__main__":
    train()
