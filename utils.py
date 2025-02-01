import torch
import heapq
import pickle
import gzip
import editdistance
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()

"""
得到的分数是每个采样句子中，在大于等于最小长度的句子中的任意一个位置截止的概率的家和做成的分数
"""
@torch.no_grad()
def score_fast(
    model,                      # base model，非微调模型
    encoded_input,              # 编码后的输入（batch），在后面使用的时候就是已经采样好的多条语句的 token id 序列
    generated_tokens,
    termination_token_id,       # 句号 token id
    min_len,                    # 最小句子长度
    skip_first,                 # 为 prompt_length
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,      
):
    # 再次获取模型输出下一个 token 的得分
    if prompt_cache is None:
        y_pred = model.llm.forward_all(encoded_input)
        logits = model.llm_decoder(y_pred)
    else:
        # NOTE: 这里我看应该用不着，所以写死了
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        raise NotImplementedError("DO NOT USE CACHE for scoring")
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits
    # 去除 prompt 部分的得分（从 prompt 最后一个概率开始）
    # get rid of the first few tokens
    logits = logits[:, skip_first - 1 :]
    # 根据词汇偏好（好与坏）进行概率补偿
    # score the log probability of the input sequence while ignoring termination and padding tokens
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        logits[:, :, ~vocab_nice_mask] += vocab_alpha
    elif vocab_naughty_mask is not None:
        # add vocab_alpha to the logits of the masked vocab items
        logits[:, :, vocab_naughty_mask] += vocab_alpha
    logits[:, :, termination_token_id+1:] = -torch.inf
    # softmax 转化成每组词汇的概率
    logprob = logits.log_softmax(-1)
    # 提取原来输入的采样后的语句中的生成部分的 token id 序列，并进行维度扩展
    token_ids = generated_tokens.unsqueeze(-1)
    # 收集之前生成的每句话的 token_ids 对应的概率的对数
    logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
    # 逐步累加每句话的采样的所有词汇的概率，即每步可以停止时，当前生成的句子的概率之和
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)，即给定 prompt 下，生成的句子的概率之和
    # 获取每句话所有词汇位置的终止标记的概率，并作为初始 reward
    reward = logprob[
        :, :, termination_token_id
    ]  # logP(generated[i+1]=term | prompt + generated[:i+1])，在i+1处停止时的概率
    # ------------------- 尝试为term token添加阈值 penalty -------------------- #
    # reward = torch.where(reward<torch.tensor([-5], dtype=reward.dtype, device=reward.device), 
    #                     reward+torch.tensor([-8000], dtype=reward.dtype, device=reward.device),
    #                     reward)
    # ------------------- 尝试为term token添加阈值 penalty -------------------- #
    # 加上之前停止时候的概率，就得到了在任意一个地方停止时整个句子的生成概率
    reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
    # ------------------- 尝试添加 baseline 、除以长度、修改温度 -------------------- #
    # logP = logP - (logP.sum(dim=0)/logP.shape[0])
    # logP = logP / torch.arange(1, logP.shape[1]+1, dtype=logP.dtype, device=logP.device).unsqueeze(0)
    # reward[:, 1:] = reward[:, 1:] / torch.arange(1, reward.shape[1], dtype=reward.dtype, device=reward.device).unsqueeze(0)
    # ------------------- 尝试添加 baseline 、除以长度、修改温度 -------------------- #
    # 标识哪些位置不是终止令牌，标识从生成的位置开始，一旦遇到终止令牌标记则标志为 false，否则为 true
    non_term_mask = (generated_tokens != termination_token_id)
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
    reward_unpenalized = reward.clone()
    # 将小于最小句子长度的句子奖励设置为 -99。
    # reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
    return reward, reward_unpenalized

@torch.no_grad()
def score_fast_confidence(
    model,                      # base model，非微调模型
    encoded_input,              # 编码后的输入（batch），在后面使用的时候就是已经采样好的多条语句的 token id 序列
    generated_tokens,
    termination_token_id,       # 句号 token id
    min_len,                    # 最小句子长度
    skip_first,                 # 为 prompt_length
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,
    rp_window_size=20,
    rp_factor=15,      
):
    # 再次获取模型输出下一个 token 的得分
    if prompt_cache is None:
        y_pred = model.llm.forward_all(encoded_input)
        logits = model.llm_decoder(y_pred)
    else:
        # NOTE: 这里我看应该用不着，所以写死了
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        raise NotImplementedError("DO NOT USE CACHE for scoring")
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits
    # 去除 prompt 部分的得分（从 prompt 最后一个概率开始）
    # get rid of the first few tokens
    logits = logits[:, skip_first - 1 :]
    # 根据词汇偏好（好与坏）进行概率补偿
    # score the log probability of the input sequence while ignoring termination and padding tokens
    if vocab_nice_mask is not None:
        # add vocab_alpha to the logits of the unmasked vocab items
        logits[:, :, ~vocab_nice_mask] += vocab_alpha
    elif vocab_naughty_mask is not None:
        # add vocab_alpha to the logits of the masked vocab items
        logits[:, :, vocab_naughty_mask] += vocab_alpha
    logits[:, :, termination_token_id+1:] = -torch.inf
    # softmax 转化成每组词汇的对数概率
    logprob = logits.softmax(-1)

    # --------------------- Repetition Penalty --------------------- #
    for t in range(1, generated_tokens.shape[1]):
        former_generated_tokens = generated_tokens[:, :t]
        if t > rp_window_size:
            last_tokens = former_generated_tokens[:, -rp_window_size:]  # [B, window]
        else:
            last_tokens = former_generated_tokens  # [B, T]
        rp_matrix = torch.ones_like(logprob[:, t, :])
        for i in range(last_tokens.size(0)):  # 遍历 batch
            rp_matrix[i, torch.unique(last_tokens[i])] = rp_factor
        
        logprob[:, t, :] = logprob[:, t, :]**rp_matrix
    # --------------------- Repetition Penalty --------------------- #

    top2_values, _ = torch.topk(logprob, k=2, dim=-1)
    max_prob = top2_values[:, :, 0]
    second_max_prob = top2_values[:, :, 1]
    reward = torch.log(max_prob - second_max_prob)
    reward = reward.cumsum(dim=-1)
    # normalization
    # reward = reward / torch.arange(1, reward.shape[1]+1, dtype=reward.dtype, device=reward.device).unsqueeze(0)
    
    # 标识哪些位置不是终止令牌，标识从生成的位置开始，一旦遇到终止令牌标记则标志为 false，否则为 true
    non_term_mask = (generated_tokens != termination_token_id)
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
    reward_unpenalized = reward.clone()
    # 将小于最小句子长度的句子奖励设置为 -99。
    # reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
    return reward, reward_unpenalized


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        temperature=1.0,
        min_len=1,
        vocab_alpha=-50.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        sentence_validator=None,
        valid_sentence_alpha=None,
    ):
        assert (
            sentence_validator is None
            and valid_sentence_alpha is None
            or sentence_validator is not None
            and valid_sentence_alpha is not None
        )
        # print(f"input temperature: {temperature}")
        self.temperature = temperature
        self.sentence_token_id = sentence_token_id
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
    
    # 计算分数函数
    def score(self, input_batch, generated_tokens, prompt_length, model):
        # 将模型从 lora 切换到 base 模式，lora模式为 low-rank adaptation
        # 这是确保在评分过程中使用的是基础模型而不是微调模型
        lora_to_base(model)
        # 保存当前训练状态并设置为评估模式
        training = model.training
        model.eval()
        # 计算奖励分数
        reward, reward_unpenalized = score_fast_confidence(
            model=model,                                    # 基础模型实例
            encoded_input=input_batch,                      # 编码后的输入（batch）
            generated_tokens=generated_tokens,               # [B,T]
            termination_token_id=self.sentence_token_id,    # 句子结束标记的 token id
            skip_first=prompt_length,                       # prompt 长度，这部分不评分，会跳过
            vocab_nice_mask=self.vocab_nice_mask,           # 词汇表掩码（nice），好像在本代码中没有指定
            vocab_naughty_mask=self.vocab_naughty_mask,     # 非法词汇表掩码
            vocab_alpha=self.vocab_alpha,                   # 词汇偏好（好与坏）概率补偿
            min_len=self.min_len,                           # 最小句子长度约束
        )
        # print(f"reward: {reward}")
        # print(f"temperature: {self.temperature}")
        reward /= self.temperature
        reward_unpenalized /= self.temperature
        # print(f"changed reward: {reward}")
        base_to_lora(model)
        if training:
            model.train()

        # NOTE: sentence_validator 无法使用
        if self.sentence_validator is not None:
            raise NotImplementedError("sentence_validator CAN NOT USE!")
        #     invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
        #     invalid = invalid * self.valid_sentence_alpha
        #     reward = torch.min(reward, invalid)

        return reward, reward_unpenalized


class SentenceValidator:
    def __init__(self, sentence_token_id) -> None:
        self.sentence_token_id = sentence_token_id

    def __call__(self, sentences, tokenizer):
        pass


class RuleSentenceValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("en_core_web_lg")

    def __call__(self, sentences, tokenizer):
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=sentences.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == self.sentence_token_id:
                    break  # Only unterminated sentences get a reward
                sent = tokenizer.decode(sentences[i, : j + 1])
                sent = self.nlp(sent).sents
                tokens = []
                for s in sent:
                    for t in s:
                        tokens.append(t)
                if not (len(tokens) >= 2 and tokens[0].is_space and tokens[1].is_title):
                    invalid[i, j + 1] = True  # Must start with a space and capital
                    continue
                has_noun = 1
                has_verb = 1
                for token in tokens:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ in ["VERB", "AUX"]:
                        has_verb -= 1
                if has_noun > 0 or has_verb > 0:
                    invalid[i, j + 1] = True  # Must have a noun and a verb
        return invalid


class ModelSentenceValidator(SentenceValidator):
    def __init__(self, *args, model_name=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if model_name is None:
            model_name = "textattack/roberta-base-CoLA"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )

    @torch.no_grad()
    def __call__(self, sentences, tokenizer):
        sentences = sentences.to(self.model.device)
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=self.model.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        done = torch.zeros(sentences.shape[0]).bool().to(self.model.device)
        for i in range(sentences.shape[1]):
            sent = sentences[:, : i + 1]
            done |= sent[:, -1] == self.sentence_token_id
            if done.all():
                break
            sent = self.tokenizer(
                tokenizer.batch_decode(sent),
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            invalid_probs = self.model(**sent).logits.softmax(dim=-1)[:, 0]
            invalid[~done, i + 1] = invalid_probs[~done] > 0.2
        return invalid


def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
):
    # 每一步都生成并返回句子的终止概率
    # generate and return the probability of terminating at every step
    # 表示哪些序列仍在生成状态，初始时所有序列为活跃状态。
    min_len = encoded_prompt["target_text_token_len"]*2
    encoded_prompt = encoded_prompt["lm_input"]
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    # 存储当前生成的状态
    state = encoded_prompt.clone()
    # 存储前向和终止的概率
    log_pf = []
    log_pterm = []
    # 存储句子的生成状态
    lm_input = state  # For caching hidden states during generation
    cache = None  # For caching hidden states during generation
    generated_tokens = []  # 用于存储生成的 token_id
    # 生成循环，生成次数为最大句子长度
    for i in range(max_len + 1):
        # 输入prompt的 token id 序列和 past key values，获取当前步的输出，这是最重要的
        y_pred, cache = model.llm.forward_one_step(lm_input, 
                                            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                            cache=cache)

        # 获取最后一层的 logits，即每个 token 的预测得分，相当于每次都往前推进一个单词（token）
        # 这个地方多了个 log_softmax, 不知道会不会有问题
        # logits = model.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        logits = model.llm_decoder(y_pred[:, -1])
        # print(logits)
        # 没有动作序列，则进行自动生成
        if action_seq is None:
            with torch.no_grad():
                # 使用 softmax 转化为概率
                prob = logits.softmax(dim=-1)
                modified_logits = logits.clone().detach()
                # 进行 top-k 采样，只保留概率最高的 K 个 token，其余 token 的概率被设置为零
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -torch.inf
                # 进行 top-p 采样，只保留概率和前 K 个 token 的和为 top_p 的 token，其余 token 的概率被设置为零
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.0:
                    sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                    nucleus = cumsum_prob < top_p
                    nucleus = torch.cat(
                        [
                            nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                            nucleus[..., :-1],
                        ],
                        dim=-1,
                    )
                    modified_logits[~nucleus] = -torch.inf
                # 如果此时还处于最小长度限制内，则将终止 token 的概率设置为无穷小
                if i < min_len:
                    # if we haven't reach the minimum length, set the probability of terminating to 0
                    modified_logits[:, termination_token_id] = -torch.inf
                # 如果此时已经达到最大长度限制，则将终止 token 的概率设置为1，其他设置为无穷小
                elif i >= max_len:
                    # if we've reached the maximum length, set the probability of terminating to 1
                    mask = [True] * modified_logits.shape[1]
                    mask[termination_token_id] = False
                    modified_logits[:, mask] = -torch.inf
                # 将非 nice 的词汇的概率设置为负数
                if vocab_nice_mask is not None:
                    # add vocab_alpha to the logits of the unmasked vocab items
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                # 将非法词汇的概率设置为负数
                if vocab_naughty_mask is not None:
                    # add vocab_alpha to the logits of the masked vocab items
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha
                # 将CosyVoice的非法词汇概率设置为负数
                modified_logits[:, termination_token_id+1:] = -torch.inf
                # 进行温度处理，让分布更尖或者更平缓
                prob = (modified_logits / temperature).softmax(dim=-1)
                # 根据概率采样下一个token，生成每一句的下一个 token id
                token_ids = torch.multinomial(prob, num_samples=1) # [B, 1]
        else: # 一般是从奖励缓冲区中进行采样就会有固定序列 
            if i >= action_seq.size(-1):
                token_ids = (
                    torch.ones_like(action_seq[:, 0]) * termination_token_id
                ).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        # 根据 active_seqs 标记，将已经终止的语句的此时的 token_id 替换为终止 token_id，后面就不再让他生成了，全加的是终止token_id
        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        # 对原来的 logits 根据对应的 mask 进行处理
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            # print(vocab_naughty_mask)
            logits[:, vocab_naughty_mask] += vocab_alpha
        # 计算对应概率
        logprob = logits.log_softmax(dim=-1)
        # 记录终止概率，只记录 alive 的样本的终止概率，死掉的这一步记为0
        # 终止概率即下一步输出终止 token 的概率
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, termination_token_id],
                0,
            )
        )
        # 更新 active_seqs 标记，如果 token_id 是终止 token id，则标记为 False
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        # 记录前向概率，只记录 alive 的样本的前向概率
        # 前向概率即本次采样选取的 token 对应的概率，死掉的这一步记为0
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        generated_tokens.append(token_ids)
        lm_input = model.speech_embedding(token_ids)
        assert lm_input.shape[0]==state.shape[0] and lm_input.shape[2]==state.shape[2], f"lm_input.shape: {lm_input.shape}, state.shape: {state.shape}"
        # 拼接 token_id 到 state 中
        state = torch.cat([state, lm_input], dim=1)
        # check if all sequences have terminated
        # 如果所有句子都结束了那就结束
        if torch.all(~active_seqs):
            break
    # 现在每个句子都已经生成完了，这里对列表中的 tensor 进行拼接，变成一整个 tensor
    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    generated_tokens = torch.cat(generated_tokens, dim=-1)  # [B,T]
    # 计算奖励，forward 中，skip_rewards=False
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # 对得到的所有采样语句的 token_id 序列计算奖励分数，剔除最后一个 token，因为这里可以确保是
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(input_batch=state[:, :-1], generated_tokens=generated_tokens[:, :-1])
    # add a termination token to the end of the sequence
    return generated_tokens, state, log_pf, log_pterm, log_r, log_r_unpenalized

def confidently_generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
    rp_window_size=20,
    rp_factor=15,
):
    # 每一步都生成并返回句子的终止概率
    # generate and return the probability of terminating at every step
    # 表示哪些序列仍在生成状态，初始时所有序列为活跃状态。
    min_len = encoded_prompt["target_text_token_len"]*2
    encoded_prompt = encoded_prompt["lm_input"]
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    # 存储当前生成的状态
    state = encoded_prompt.clone()
    # 存储前向和终止的概率
    log_pf = []
    log_pterm = []
    # 存储句子的生成状态
    lm_input = state  # For caching hidden states during generation
    cache = None  # For caching hidden states during generation
    generated_tokens = []  # 用于存储生成的 token_id
    # 生成循环，生成次数为最大句子长度
    for i in range(max_len + 1):
        # 输入prompt的 token id 序列和 past key values，获取当前步的输出，这是最重要的
        y_pred, cache = model.llm.forward_one_step(lm_input, 
                                            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                            cache=cache)

        # 获取最后一层的 logits，即每个 token 的预测得分，相当于每次都往前推进一个单词（token）
        # 这个地方多了个 log_softmax, 不知道会不会有问题
        # logits = model.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        logits = model.llm_decoder(y_pred[:, -1])
        # print(logits)
        # 没有动作序列，则进行自动生成
        if action_seq is None:
            with torch.no_grad():
                # 使用 softmax 转化为概率
                prob = logits.softmax(dim=-1)
                modified_logits = logits.clone().detach()
                # 进行 top-k 采样，只保留概率最高的 K 个 token，其余 token 的概率被设置为零
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -torch.inf
                # 进行 top-p 采样，只保留概率和前 K 个 token 的和为 top_p 的 token，其余 token 的概率被设置为零
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.0:
                    sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                    nucleus = cumsum_prob < top_p
                    nucleus = torch.cat(
                        [
                            nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                            nucleus[..., :-1],
                        ],
                        dim=-1,
                    )
                    modified_logits[~nucleus] = -torch.inf
                # 如果此时还处于最小长度限制内，则将终止 token 的概率设置为无穷小
                if i < min_len:
                    # if we haven't reach the minimum length, set the probability of terminating to 0
                    modified_logits[:, termination_token_id] = -torch.inf
                # 如果此时已经达到最大长度限制，则将终止 token 的概率设置为1，其他设置为无穷小
                elif i >= max_len:
                    # if we've reached the maximum length, set the probability of terminating to 1
                    mask = [True] * modified_logits.shape[1]
                    mask[termination_token_id] = False
                    modified_logits[:, mask] = -torch.inf
                # 将非 nice 的词汇的概率设置为负数
                if vocab_nice_mask is not None:
                    # add vocab_alpha to the logits of the unmasked vocab items
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                # 将非法词汇的概率设置为负数
                if vocab_naughty_mask is not None:
                    # add vocab_alpha to the logits of the masked vocab items
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha
                # 将CosyVoice的非法词汇概率设置为负数
                modified_logits[:, termination_token_id+1:] = -torch.inf
                # ----------------- Repeat Penalty ---------------------- #
                if i != 0:
                    if len(generated_tokens) > rp_window_size:
                        last_tokens = torch.cat(generated_tokens[-rp_window_size:], dim=-1)  # [B, window]
                    else:
                        last_tokens = torch.cat(generated_tokens, dim=-1)  # [B, T]
                    rp_matrix = torch.ones_like(modified_logits)
                    for i in range(last_tokens.size(0)):  # 遍历 batch
                        rp_matrix[i, torch.unique(last_tokens[i])] = rp_factor
                    
                    modified_logits = modified_logits**rp_matrix
                # ----------------- Repeat Penalty ---------------------- #
                # 这里将不再进行温度处理，因为会导致不确定性和训练困难
                prob = (modified_logits).softmax(dim=-1)
                # 根据概率使用贪心采样获取下一个token id，并在第一个 token 处做多样化处理
                if i == 0:
                    token_ids = torch.diag(torch.topk(prob, k=prob.shape[0], dim=-1)[1]).unsqueeze(-1)
                else:
                    token_ids = torch.argmax(prob, dim=-1, keepdim=True) # [B, 1]
        else: # 一般是从奖励缓冲区中进行采样就会有固定序列 
            if i >= action_seq.size(-1):
                token_ids = (
                    torch.ones_like(action_seq[:, 0]) * termination_token_id
                ).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        # 根据 active_seqs 标记，将已经终止的语句的此时的 token_id 替换为终止 token_id，后面就不再让他生成了，全加的是终止token_id
        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        # 对原来的 logits 根据对应的 mask 进行处理
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            # print(vocab_naughty_mask)
            logits[:, vocab_naughty_mask] += vocab_alpha
        # 计算对应概率
        logprob = logits.log_softmax(dim=-1)
        # ----------------- Repeat Penalty ---------------------- #
        # 这里因为已经是 log_softmax 后的，因此我们做乘法惩罚
        if i != 0:
            if len(generated_tokens) > rp_window_size:
                last_tokens = torch.cat(generated_tokens[-rp_window_size:], dim=-1)  # [B, window]
            else:
                last_tokens = torch.cat(generated_tokens, dim=-1)  # [B, T]
            rp_matrix = torch.ones_like(modified_logits)
            for i in range(last_tokens.size(0)):  # 遍历 batch
                rp_matrix[i, torch.unique(last_tokens[i])] = rp_factor  # 去重并施加 penalty
            
            logprob = logprob*rp_matrix
        # ----------------- Repeat Penalty ---------------------- #
        # 记录终止概率，只记录 alive 的样本的终止概率，死掉的这一步记为0
        # 终止概率即下一步输出终止 token 的概率
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, termination_token_id],
                0,
            )
        )
        # 更新 active_seqs 标记，如果 token_id 是终止 token id，则标记为 False
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        # 记录前向概率，只记录 alive 的样本的前向概率
        # 前向概率即本次采样选取的 token 对应的概率，死掉的这一步记为0
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        generated_tokens.append(token_ids)
        lm_input = model.speech_embedding(token_ids)
        assert lm_input.shape[0]==state.shape[0] and lm_input.shape[2]==state.shape[2], f"lm_input.shape: {lm_input.shape}, state.shape: {state.shape}"
        # 拼接 token_id 到 state 中
        state = torch.cat([state, lm_input], dim=1)
        # check if all sequences have terminated
        # 如果所有句子都结束了那就结束
        if torch.all(~active_seqs):
            break
    # 现在每个句子都已经生成完了，这里对列表中的 tensor 进行拼接，变成一整个 tensor
    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    generated_tokens = torch.cat(generated_tokens, dim=-1)  # [B,T]
    # 计算奖励，forward 中，skip_rewards=False
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # 对得到的所有采样语句的 token_id 序列计算奖励分数，剔除最后一个 token，因为这里可以确保是
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(input_batch=state[:, :-1], generated_tokens=generated_tokens[:, :-1])
    # add a termination token to the end of the sequence
    return generated_tokens, state, log_pf, log_pterm, log_r, log_r_unpenalized

# 本函数为 GFN 的子轨迹平衡损失
def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1]
    ), f"log_pf.shape: {log_pf.shape}, log_r.shape: {log_r.shape}, log_pterm.shape: {log_pterm.shape}, generated_text.shape: {generated_text.shape}"
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, :-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1]
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda

    return batch_loss

def trajectory_balance_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1]
    ), f"log_pf.shape: {log_pf.shape}, log_r.shape: {log_r.shape}, log_pterm.shape: {log_pterm.shape}, generated_text.shape: {generated_text.shape}"
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    # delta = (
    #     log_r[:, :-1]
    #     + log_pf[:, :-1]
    #     + log_pterm[:, 1:]
    #     - log_r[:, 1:]
    #     - log_pterm[:, :-1]
    # )
    # print(f"log_r: {log_r}")
    # print(f"log_pf: {log_pf}")
    # print(f"log_pterm: {log_pterm}")
    # delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)
    log_r_last = log_r.gather(1, (log_r!=0).cumsum(1).argmax(1).unsqueeze(1))
    log_pterm_last = log_pterm.gather(1, (log_pterm!=0).cumsum(1).argmax(1).unsqueeze(1))
    # log_pf_end_idx = (log_pf!=0).cumsum(1).argmax(1).unsqueeze(1)
    log_pf_sum = log_pf.cumsum(1)[:, -1].unsqueeze(1)
    # log_pf_sum /= (log_pf_end_idx+1)
    
    # print(f"log_r_last: {log_r_last}")
    # print(f"log_pterm_last: {log_pterm_last}")
    # print(f"log_pf_sum: {log_pf_sum}")
    # Get a mask for tokens after the termination token in the generated_text
    # 其中，已经结束的为 true
    # 刚结束和还未结束的为 false
    # mask = (generated_text[:, :-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1]
    subtraj_len = generated_len-1
    subtb_term = (
        # delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        log_r[:, 0].unsqueeze(1) + log_pf_sum + log_pterm_last - log_r_last - log_pterm[:, 0].unsqueeze(1)
        # log_pf_sum + log_pterm_last - log_r_last
    ) ** 2
    # print(f"delta_cumsum[:, subtraj_len:]: {delta_cumsum[:, subtraj_len:]}, delta_cumsum[:, :-subtraj_len]: {delta_cumsum[:, :-subtraj_len]}")
    # print(f"subtb_term: {subtb_term}")
    # subtb_term[mask[:, subtraj_len - 1 :]] = 0
    # print(f"changed subtb_term: {subtb_term}")
    batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
    total_lambda += (
        # subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        generated_text.shape[0]
    )
    # print(f"batch_loss: {batch_loss}, total_lambda: {total_lambda}")
    batch_loss /= total_lambda

    return batch_loss

def trajectory_balance_confidence_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1]
    ), f"log_pf.shape: {log_pf.shape}, log_r.shape: {log_r.shape}, log_pterm.shape: {log_pterm.shape}, generated_text.shape: {generated_text.shape}"
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    # delta = (
    #     log_r[:, :-1]
    #     + log_pf[:, :-1]
    #     + log_pterm[:, 1:]
    #     - log_r[:, 1:]
    #     - log_pterm[:, :-1]
    # )
    # print(f"log_r: {log_r}")
    # print(f"log_pf: {log_pf}")
    # print(f"log_pterm: {log_pterm}")
    # delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)
    log_r_last = log_r.gather(1, (log_r!=0).cumsum(1).argmax(1).unsqueeze(1)) 
    log_pterm_last = log_pterm.gather(1, (log_pterm!=0).cumsum(1).argmax(1).unsqueeze(1))
    # log_pf_end_idx = (log_pf!=0).cumsum(1).argmax(1).unsqueeze(1)
    log_pf_sum = log_pf.cumsum(1)[:, -1].unsqueeze(1)
    # log_pf_sum /= (log_pf_end_idx+1)
    
    # print(f"log_r_last: {log_r_last}")
    # print(f"log_pterm_last: {log_pterm_last}")
    # print(f"log_pf_sum: {log_pf_sum}")
    # Get a mask for tokens after the termination token in the generated_text
    # 其中，已经结束的为 true
    # 刚结束和还未结束的为 false
    # mask = (generated_text[:, :-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1]
    subtraj_len = generated_len-1
    subtb_term = (
        # delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        # log_r[:, 0].unsqueeze(1) + log_pf_sum + log_pterm_last - log_r_last - log_pterm[:, 0].unsqueeze(1)
        log_pf_sum + log_pterm_last - log_r_last
    ) ** 2
    # print(f"delta_cumsum[:, subtraj_len:]: {delta_cumsum[:, subtraj_len:]}, delta_cumsum[:, :-subtraj_len]: {delta_cumsum[:, :-subtraj_len]}")
    # print(f"subtb_term: {subtb_term}")
    # subtb_term[mask[:, subtraj_len - 1 :]] = 0
    # print(f"changed subtb_term: {subtb_term}")
    batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
    total_lambda += (
        # subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        generated_text.shape[0]
    )
    # print(f"batch_loss: {batch_loss}, total_lambda: {total_lambda}")
    batch_loss /= total_lambda

    return batch_loss

# 本函数为去掉中间的termination token reward 的 GFN 的子轨迹平衡损失
def modified_subtb_loss_without_eos_reward(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1]
    ), f"log_pf.shape: {log_pf.shape}, log_r.shape: {log_r.shape}, log_pterm.shape: {log_pterm.shape}, generated_text.shape: {generated_text.shape}"
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        # + log_pterm[:, 1:]
        - log_r[:, 1:]
        # - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, :-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1]
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda
    # batch_loss += ((log_pterm.gather(1, (log_pterm!=0).cumsum(1).argmax(1).unsqueeze(1)))**2).sum()/generated_text.shape[0]

    return batch_loss

# 用于计算生成文本的终止位置相关的值，包括累积的前向概率 (log_pfs)、奖励 (log_r) 和未惩罚的奖励 (log_r_unpenalized)
def get_termination_vals(
    generated_text,
    log_pf,
    log_pterm,
    log_r,
    log_r_unpenalized,
    termination_token_id,
    prompt_len,
):
    # batch idx 为每个生成的文本的索引
    batch_idx = torch.arange(generated_text.size(0))
    # 获取每个文本生成部分的长度（不包含终止标记）
    gen_len = (
        (generated_text == termination_token_id).byte().argmax(dim=-1)
    )
    if log_pf is None and log_pterm is None:
        log_pfs = None
    else:
        # 在 log_pf 中插入一个 0 列，用于计算 cumsum，顺便删除最后一列
        log_pf = torch.cat([torch.zeros_like(log_pf[:, :1]), log_pf], dim=-1)[:, :-1]
        # 计算累积前向概率，并添加当前位置终止的概率
        log_pfs = log_pf.cumsum(dim=-1) + log_pterm
        # 得到各个生成的语句的累计前向概率加上位置终止的概率
        log_pfs = log_pfs[batch_idx, gen_len]
    # 获取各个生成语句的直接奖励分数和未惩罚奖励分数
    log_r = log_r[batch_idx, gen_len]
    log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    return log_pfs, log_r, log_r_unpenalized, gen_len


class SequenceDiversity:
    def __init__(self, method, **kwargs):
        self.method = method
        if method is None:
            pass
        # 本次训练采用这个方案
        elif method == "sequence_embedding":
            # 获取 model name 参数，默认为后面的内容
            model_name = kwargs.get(
                "model_name", "sentence-transformers/all-mpnet-base-v2"
            )
            # 设置 model
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    # 调用当前类的方法如下，接收一个 sequences 参数，应该是多个句子的 token_id 序列
    @torch.no_grad()
    def __call__(self, sequences):
        if self.method is None:
            return None
        elif self.method == "sequence_embedding":
            # 通过编码器获取输入语句的 embeddings
            embeddings = self.model.encode(sequences, show_progress_bar=False)
            # 计算 cosine similarity
            sim = cos_sim(embeddings, embeddings)
            # 使用 torch.triu_indices 获取上三角矩阵的索引（排除对角线），以避免重复计算
            indices = torch.triu_indices(len(sequences), len(sequences), offset=1)
            # 计算非对角线上元素的平均值，并从1中减去得到多样性分数。
            diversity = 1 - sim[indices[0], indices[1]].mean().item()
        else:
            raise ValueError(f"Unknown sequence diversity method: {self.method}")
        return diversity


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    """
    item 是一个字典，包含：
        "logreward": 当前存储的生成语句的奖励。
        "str_sentence": 生成句子的字符串表示。
        "tensor_sentence": 生成句子的张量表示。
        "full_logrewards": 完整的生成语句，包含子轨迹的对数奖励序列。

    self._buffer 是一个字典，key 是提示字符串 str_prompt，value 是另一个字典，包含：
        "tensor_prompt": 提示的 token 表示。
        "sentences": 一个列表，包含所有这个提示相关的 item。
        "exists": 一个集合，包含所有与这个提示相关的 item 的生成句子的字符串。
    """
    def __init__(self, buffer_size, termination_token_id, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.sim_tolerance = sim_tolerance                  # 相似度容忍阈值
        self.reset()

    # 清空缓存
    def reset(self):
        self._buffer = {}

    # 添加一个 item 到缓存中
    def add(self, item):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # 如果这个 item 对应的生成的字符串已经在 buffer 中，则跳过不进行添加操作
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        # 计算新项目与缓冲区中每个项目的编辑距离。
        tokenized_sentence = [
            x
            for x in item["tensor_sentence"].tolist()
            if x != self.termination_token_id
        ]
        for buffer_item in self._buffer[str_prompt]["sentences"]:
            tokenized_existing_sentence = [
                x for x in buffer_item[2].tolist() if x != self.termination_token_id
            ]
            # 如果遍历的当前项目的编辑距离小于阈值
            if (
                editdistance.eval(tokenized_sentence, tokenized_existing_sentence)
                < (len(tokenized_sentence) + len(tokenized_existing_sentence))
                * self.sim_tolerance
            ):
                # 当前项目的日志奖励大于等于新项目的日志奖励，则跳过新项目。
                if buffer_item[0] >= item["logreward"]:
                    return
                # 新项目的日志奖励大于当前项目的日志奖励，则删除旧项目，并添加新的项目
                else:
                    self._buffer[str_prompt]["exists"].remove(buffer_item[1])
                    self._buffer[str_prompt]["sentences"].remove(buffer_item)
                    heapq.heapify(self._buffer[str_prompt]["sentences"])
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                    heapq.heappush(
                        self._buffer[str_prompt]["sentences"],
                        (
                            item["logreward"],
                            item["str_sentence"],
                            item["tensor_sentence"],
                            item["full_logrewards"],
                        ),
                    )
                    return
        # 已确定添加，注册当前语句，防止后面存在重复计算
        self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        # 如果缓存已满，则弹出最小奖励的项目并添加新项目
        if len(self._buffer[str_prompt]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )
            self._buffer[str_prompt]["exists"].remove(popped[1])
        # 否则直接添加新的项目
        else:
            heapq.heappush(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )

    # 批量添加新的项目（包含新的提示生成集合）
    def add_batch(self, prompt, sentences, logrewards):
        """
        add a batch of items to the buffer
        """
        def batch_decode(sentences):
            token_sentences = []
            for sentence in sentences:
                name = " ".join([str(x) for x in sentence.tolist()])
                name = name.replace(str(self.termination_token_id), "").strip()
                token_sentences.append(name)
            return token_sentences
        
        str_prompt = " ".join([str(x) for x in prompt[:, 0].tolist()])
        # 如果这个提示字符串是新的字符串，则直接创建一个条目
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        # 将终止 token 之后的所有 token 设置为终止 token 。
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        # 解码出句子字符串
        token_sentences = batch_decode(sentences)
        # 逐个添加对应的句子（会将句号删除）
        for i in range(sentences.size(0)):
            str_sentence = token_sentences[i].strip()
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                }
            )

    # 从缓冲区中均匀采样一批项目，并返回堆叠的张量
    def sample(self, batch_size, prompt):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        # 将提示张量转为字符串形式
        str_prompt = " ".join([str(x) for x in prompt[:, 0].tolist()])
        # 提示张量不在缓冲区，则返回 None
        if str_prompt not in self._buffer:
            return None, None
        # 从缓冲区中随机采样一批项目
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        # 抽取 batch_size 个随机 idx，允许重复
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        # 返回采样的 tensor_sentence 和 full_logrewards，使用终止 token 和 0 进行填充
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    # 打印缓冲区内容
    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    # 保存缓冲区内容
    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
