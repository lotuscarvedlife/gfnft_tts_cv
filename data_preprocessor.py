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
from tqdm import tqdm


@hydra.main(version_base=None, config_path="./configs/", config_name="train")
def data_preprocess(config: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = get_model(config)
    model.to(device)

    model_dir = config.task.model.model_dir
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
    
    base_dir = "data/"
    prompt_file_path = base_dir+"filtered_sampled_output_2.txt"
    target_file_path = base_dir+"filtered_target_output_2.txt"
    sample_rate= 24000
    save_dir = base_dir+"lm_input_tensor/"

    with open(prompt_file_path, 'r', encoding='utf-8') as prompt_file, open(target_file_path, 'r', encoding='utf-8') as target_file:
        for prompt_line, target_line in tqdm(zip(prompt_file, target_file), total=1000, desc="Processing llm input data"):
            
            # prompt 处理
            audio_file_name, prompt_text = prompt_line.split(' ', 1)
            audio_file_path = base_dir + "audio/" + audio_file_name + ".wav"
            prompt_text = prompt_text.strip()
            prompt_speech_16k = load_wav(audio_file_path, 16000)

            # target 文本处理
            tts_text = target_line.strip()

            prompt_text = frontend.text_normalize(prompt_text, split=False, text_frontend=True)
            tts_text = frontend.text_normalize(tts_text, split=False, text_frontend=True)

            model_input = frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k, sample_rate)
            text = model_input["text"].to(device)
            text_len = torch.tensor([text.shape[1]], dtype=torch.int32).to(text.device)
            prompt_text = model_input["prompt_text"].to(device)
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(prompt_text.device)
            llm_prompt_speech_token = model_input["llm_prompt_speech_token"].to(device)
            prompt_speech_token = llm_prompt_speech_token
            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(prompt_speech_token.device)
            llm_embedding = model_input["llm_embedding"].to(device)
            embedding = llm_embedding

            # 输入衔接与处理(llm.inference)
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
            text = model.llm.model.model.embed_tokens(text)
            embedding = torch.zeros(1, 0, model.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)
            sos_eos_emb = model.llm_embedding.weight[model.sos_eos].reshape(1, 1, -1)
            task_id_emb = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)
            if prompt_speech_token_len != 0:
                prompt_speech_token_emb = model.speech_embedding(prompt_speech_token)
            else:
                prompt_speech_token_emb = torch.zeros(1, 0, model.llm_input_size, dtype=text.dtype).to(device)
            lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

            torch.save(lm_input, save_dir+f"data_{audio_file_name}.pt")





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


    model_dir = config.task.model.model_dir
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    
    state_dict = torch.load('{}/llm.pt'.format(model_dir), map_location='cuda')
    model = configs["llm"]
    model.load_state_dict(state_dict, strict=False)


    return model


if __name__ == "__main__":
    data_preprocess()