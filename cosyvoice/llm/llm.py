# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        if self.fp16 is True:
            embedding = embedding.half()

        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache
    
    def forward_all(self, xs):
        outs = self.model(
            inputs_embeds=xs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            # past_key_values=cache,
        )
        # print("we have entered into forward all function")
        xs = outs.hidden_states[-1]
        # new_cache = outs.past_key_values
        return xs


class Qwen2LM(torch.nn.Module):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            vocab_alpha = -50,
            use_lora_sampling = False,
            use_lora_model = False,
            vocab_naughty = None,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        prompt_len = lm_input.shape[1]

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 4.5 add vocab_naughty_mask = [1950, 4137, 2031]
        vocab_naughty_mask = None
        if vocab_naughty is not None:
            vocab_naughty_mask = torch.zeros(self.speech_token_size+3, dtype=torch.bool)
            vocab_naughty_mask[vocab_naughty] = True

        # 5. step by step decode
        out_tokens = []
        state = lm_input.clone()
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            if use_lora_sampling == False:
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                # ---------------- 如果想要使用直接手段不采样 termination token 的话 -------------#
                # if i < min_len:
                #     logp[:, self.speech_token_size] = -torch.inf
                # ---------------- 如果想要使用直接手段不采样 termination token 的话 -------------#
                top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            else:
                # ---------------- GFN Sampling ----------------- #
                temperature = 1.0  # 在gfnft中，val 数据集采样里，不设置温度
                logp = self.llm_decoder(y_pred[:, -1])
                prob = logp.softmax(dim=-1)
                modified_logits = logp.clone().detach()
                assert modified_logits.shape[0]==1 and modified_logits.dim()==2
                # print(f"end token prob: {modified_logits[:, self.speech_token_size].item()}")
                # 如果此时还处于最小长度限制内，则将终止 token 的概率设置为无穷小
                if i < min_len:
                    # if we haven't reach the minimum length, set the probability of terminating to 0
                    modified_logits[:, self.speech_token_size] = -torch.inf
                # 如果此时已经达到最大长度限制，则将终止 token 的概率设置为1，其他设置为无穷小
                elif i >= max_len-1:
                    # if we've reached the maximum length, set the probability of terminating to 1
                    mask = [True] * modified_logits.shape[1]
                    mask[self.speech_token_size] = False
                    modified_logits[:, mask] = -torch.inf
                # 将CosyVoice的非法词汇概率设置为负数
                modified_logits[:, self.speech_token_size+1:] = -torch.inf
                if vocab_naughty_mask is not None:
                    # print(f"modified_logits.shape: {modified_logits.shape}")
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha
                # 进行温度处理，让分布更尖或者更平缓
                prob = (modified_logits / temperature).softmax(dim=-1)
                # 根据概率采样下一个token，生成每一句的下一个 token id
                top_ids = torch.multinomial(prob.squeeze(dim=0), num_samples=1).item()
                # print(f"chosen token prob: {modified_logits[:, top_ids].item()}")
                # ---------------- GFN Sampling ----------------- #
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            state = torch.cat([state, lm_input], dim=1)

        if use_lora_model==False:
            # ------------------ 计算得分模块 -------------------- #
            # print(out_tokens)
            # reward_temperature = 0.005
            out_tokens = torch.tensor(out_tokens[:-1], device=device).unsqueeze(0)
            state = state[:, :-1].to(device)
            assert state.shape[0] == 1, state.shape
            y_pred, _= self.llm.forward_one_step(state, 
                                                masks=torch.tril(torch.ones((1, state.shape[1], state.shape[1]), 
                                                                            device=device)).to(torch.bool)
                                                )
            logits = self.llm_decoder(y_pred)
            # 去除 prompt 部分的得分（从 prompt 最后一个概率开始）
            # get rid of the first few tokens
            logits = logits[:, prompt_len - 1 :]
            # softmax 转化成每组词汇的概率
            logprob = logits.log_softmax(-1)
            # 提取原来输入的采样后的语句中的生成部分的 token id 序列，并进行维度扩展
            token_ids = out_tokens.unsqueeze(-1)
            # 收集之前生成的每句话的 token_ids 对应的概率的对数
            logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
            print(f"chosen token reward(logPF): {logPF}")
            print(f"chosen token ids: {out_tokens}")
            # 逐步累加每句话的采样的所有词汇的概率，即每步可以停止时，当前生成的句子的概率之和
            logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)，即给定 prompt 下，生成的句子的概率之和
            # logP /= reward_temperature
            # ------------------- 尝试添加 baseline -------------------- #
            # logP = logP - (logP.sum(dim=0)/logP.shape[0])
            # logP = logP / torch.arange(1, logP.shape[1]+1, dtype=logP.dtype, device=logP.device).unsqueeze(0)
            # ------------------- 尝试添加 baseline -------------------- #
            # print(f"chosen token cumsum(logP): {logP}")
            # 获取每句话所有词汇位置的终止标记的概率，并作为初始 reward
            reward = logprob[
                :, :, self.speech_token_size
            ]  # logP(generated[i+1]=term | prompt + generated[:i+1])，在i+1处停止时的概率
            # reward /= reward_temperature
            print(f"end token reward: {reward}")
            # 加上之前停止时候的概率，就得到了在任意一个地方停止时整个句子的生成概率
            reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
            # 标识哪些位置不是终止令牌，标识从生成的位置开始，一旦遇到终止令牌标记则标志为 false，否则为 true
            non_term_mask = (out_tokens != self.speech_token_size)
            # 在每一段句子中的最开始添加一个 true（即添加一列 true）
            non_term_mask = torch.cat(
                (
                    non_term_mask.new_ones(non_term_mask.shape[0], 1),
                    non_term_mask,
                ),
                dim=-1,
            )  # Start (i.e., empty) state has never terminated（即还未生成任何东西一定不是终止符）
            # 将实际中的终止标记位置后续的奖励都设置为 0
            reward[~non_term_mask] = 0.0
            # 将小于最小句子长度的句子奖励设置为 -99，防止被选择。
            min_len = 1
            reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)

            # reward /= reward_temperature
            print("This sentence got reward of")
            print(reward)
            print(f"end reward = {reward[0, -1]}")

    @torch.inference_mode()
    def inference_confidently(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            # vocab_alpha = -50,
            # use_lora_sampling = False,
            # use_lora_model = False,
            # vocab_naughty = None,
            # use_repetition_penalty = False,
            rp_window_size = 20,
            rp_factor = 15,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        prompt_len = lm_input.shape[1]

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        state = lm_input.clone()
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)

            # ---------------- GFN Sampling ----------------- #
            temperature = 1.0  # 在gfnft中，val 数据集采样里，不设置温度
            logp = self.llm_decoder(y_pred[:, -1])
            prob = logp.softmax(dim=-1)
            modified_logits = logp.clone().detach()
            assert modified_logits.shape[0]==1 and modified_logits.dim()==2
            # print(f"end token prob: {modified_logits[:, self.speech_token_size].item()}")
            # 如果此时还处于最小长度限制内，则将终止 token 的概率设置为无穷小
            if i < min_len:
                # if we haven't reach the minimum length, set the probability of terminating to 0
                modified_logits[:, self.speech_token_size] = -torch.inf
            # 如果此时已经达到最大长度限制，则将终止 token 的概率设置为1，其他设置为无穷小
            elif i >= max_len-1:
                # if we've reached the maximum length, set the probability of terminating to 1
                mask = [True] * modified_logits.shape[1]
                mask[self.speech_token_size] = False
                modified_logits[:, mask] = -torch.inf
            # 将CosyVoice的非法词汇概率设置为负数
            modified_logits[:, self.speech_token_size+1:] = -torch.inf
            last_tokens=None
            # 进行温度处理，让分布更尖或者更平缓
            prob = (modified_logits / temperature).softmax(dim=-1)
            if i != 0:
                # ----------------- Repeat Penalty ---------------------- #
                if len(out_tokens) > rp_window_size:
                    last_tokens = torch.tensor(out_tokens[-rp_window_size:])  # [window]
                else:
                    last_tokens = torch.tensor(out_tokens)  # [T]
                rp_matrix = torch.ones_like(prob)
                rp_matrix[0, torch.unique(last_tokens)] = rp_factor
                # print(f"last_tokens: {last_tokens}")
                # print(f"rp_matrix: {rp_matrix}")
                prob = prob**rp_matrix
                # print(f"modified_last_tokens_logits: {modified_logits[0, torch.unique(last_tokens)]}")
                # ----------------- Repeat Penalty ---------------------- #
            # # 根据概率采样下一个token，生成每一句的下一个 token id
            if i==0:
                top_ids = torch.multinomial(prob.squeeze(dim=0), num_samples=1).item()
            else:
                # print(f"prob[0, torch.unique(last_tokens)]: {prob[0, torch.unique(last_tokens)]}")
                top_ids = torch.argmax(prob, dim=-1).item()
            # print(f"chosen token prob: {modified_logits[:, top_ids].item()}")
            # ---------------- GFN Sampling ----------------- #
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            state = torch.cat([state, lm_input], dim=1)


    @torch.inference_mode()
    def cal_lora_model_reward(self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            out_tokens: torch.Tensor,
            vocab_alpha = -50,
            vocab_naughty = None,
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)
        prompt_len = lm_input.shape[1]

        # 4.5 add vocab_naughty_mask = [1950, 4137, 2031]
        vocab_naughty_mask = None
        if vocab_naughty is not None:
            vocab_naughty_mask = torch.zeros(self.speech_token_size+3, dtype=torch.bool)
            vocab_naughty_mask[vocab_naughty] = True
        # print("666666666666666666666666666666666")

        # 5. step by step decode
        # out_tokens = []
        state = lm_input.clone()
        cache = None
        lm_input = self.speech_embedding(out_tokens.unsqueeze(0))
        state = torch.cat([state, lm_input], dim=1)

        out_tokens = out_tokens.unsqueeze(0)
        assert state.shape[0] == 1, state.shape
        y_pred, _= self.llm.forward_one_step(state, 
                                            masks=torch.tril(torch.ones((1, state.shape[1], state.shape[1]), 
                                                                        device=device)).to(torch.bool)
                                            )
        logits = self.llm_decoder(y_pred)
        # 去除 prompt 部分的得分（从 prompt 最后一个概率开始）
        # get rid of the first few tokens
        logits = logits[:, prompt_len - 1 :]
        # vocab_naughty_mask = [1950, 4137, 2031]
        if vocab_naughty_mask is not None:
            logits[:, :, vocab_naughty_mask] += vocab_alpha
        # softmax 转化成每组词汇的概率
        logprob = logits.log_softmax(-1)
        # 提取原来输入的采样后的语句中的生成部分的 token id 序列，并进行维度扩展
        token_ids = out_tokens.unsqueeze(-1)
        # 收集之前生成的每句话的 token_ids 对应的概率的对数
        logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
        print(f"chosen token reward(logPF): {logPF}")
        print(f"chosen token ids: {out_tokens}")
        # 逐步累加每句话的采样的所有词汇的概率，即每步可以停止时，当前生成的句子的概率之和
        logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)，即给定 prompt 下，生成的句子的概率之和
        # logP /= reward_temperature
        # ------------------- 尝试添加 baseline -------------------- #
        # logP = logP - (logP.sum(dim=0)/logP.shape[0])
        # logP = logP / torch.arange(1, logP.shape[1]+1, dtype=logP.dtype, device=logP.device).unsqueeze(0)
        # ------------------- 尝试添加 baseline -------------------- #
        # print(f"chosen token cumsum(logP): {logP}")
        # 获取每句话所有词汇位置的终止标记的概率，并作为初始 reward
        reward = logprob[
            :, :, self.speech_token_size
        ]  # logP(generated[i+1]=term | prompt + generated[:i+1])，在i+1处停止时的概率
        # reward /= reward_temperature
        print(f"end token reward: {reward}")
        # 加上之前停止时候的概率，就得到了在任意一个地方停止时整个句子的生成概率
        reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
        # 标识哪些位置不是终止令牌，标识从生成的位置开始，一旦遇到终止令牌标记则标志为 false，否则为 true
        non_term_mask = (out_tokens != self.speech_token_size)
        # 在每一段句子中的最开始添加一个 true（即添加一列 true）
        non_term_mask = torch.cat(
            (
                non_term_mask.new_ones(non_term_mask.shape[0], 1),
                non_term_mask,
            ),
            dim=-1,
        )  # Start (i.e., empty) state has never terminated（即还未生成任何东西一定不是终止符）
        # 将实际中的终止标记位置后续的奖励都设置为 0
        reward[~non_term_mask] = 0.0
        # 将小于最小句子长度的句子奖励设置为 -99，防止被选择。
        # min_len = 1
        # reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)

        # reward /= reward_temperature
        print("This sentence got reward of")
        print(reward)
        print(f"end reward = {reward[0, -1]}")
