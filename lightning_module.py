import random
from functools import partial
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningModule
from utils import (
    generate_and_return_termination_logprob,
    confidently_generate_and_return_termination_logprob,
    generate_and_return_termination_logprob_repetition_penalty,
    modified_subtb_loss,
    trajectory_balance_loss,
    modified_subtb_loss_without_eos_reward,
    trajectory_balance_confidence_loss,
    get_termination_vals,
    SequenceDiversity,
)
from utils import lora_to_base, base_to_lora

# LightningModule 继承，为了模型的训练和测试等缓解而设计
class NextSentenceGFNTask(LightningModule):
    def __init__(
        self,
        model,
        # tokenizer,
        reward,
        reward_buffer,
        n_samples,
        lr,
        subtb_lambda,
        pf_temp_high,
        pf_temp_low,
        pf_temp_prob,
        use_buffer_prob,
        min_sentence_len,
        max_sentence_len,
        reward_temp_start,
        reward_temp_end,
        reward_temp_horizon,
        illegal_token_mask,
        devices_count=1,
        train_probes=None,
        val_probes=None,
        diversity_metric=None,
        use_4bit=False,
    ):
        super().__init__()
        # 忽略 model 本体和 tokenizer 的参数，保存其他（应该是微调）的超参数
        self.save_hyperparameters(ignore=["model"])

        # 指定 model、tokenizer、奖励函数、奖励缓存
        self.model = model
        # self.tokenizer = tokenizer
        self.reward = reward
        self.reward_buffer = reward_buffer

        # 指定 diversity_metric，用于计算句子多样性的评价指标，包括名字和本体
        # 在本次训练的设定中为config.task.eval.diversity_metric = "sequence_embedding"
        # NOTE: 这里设置为了 None
        # self.diversity_metric_name = f"diversity ({diversity_metric})"
        # self.diversity_metric = SequenceDiversity(diversity_metric)
        self.diversity_metric_name = None
        self.diversity_metric = None

        # 可以*self.hparams.devices_count
        # 定义一个函数，用于根据训练步数计算学习率，并返回当前学习率
        self.get_lr_at_step = lambda step: min((step) / 20 * lr, lr)
        # 定义函数用以获取当前奖励的温度值，温度随步数变化，注意运行命令中有指定参数 temp_end
        # 奖励温度从 reward_temp_start1.0 开始逐渐增加到 reward_temp_end0.8，在 reward_temp_horizon 步内完成变化
        self.get_reward_temp_at_step = lambda step: reward_temp_start + (
            reward_temp_end - reward_temp_start
        ) * min(1, (step) / reward_temp_horizon)

        # # 获取一个句子结束符的 token id
        self.end_of_sentence_token_id = model.speech_token_size
        # try:  # Some tokenizers encode a "." differently when it is the first token
        #     self.end_of_sentence_token_id = tokenizer.encode(
        #         "A sentence.", add_special_tokens=False
        #     )[-1]
        # except:
        #     self.end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(".")

    """ 
    推理函数，定义模型前向传播过程
    prompt：提示
    n_samples：一个提示生成候选句子的数量，默认为1，设置里面是 20
    pf_temperature：采样温度，默认为1.0
    action_seq：用于指定生成句子的策略，用于指导生成过程，默认为None
    """
    def forward(self, prompt, n_samples=None, pf_temperature=1.0, action_seq=None):
        # 保证 prompt 只有一句话
        assert prompt["lm_input"].ndim == 2
        assert isinstance(prompt, dict)
        # 设置采样的样本数量，默认使用超参数中的设置（20）
        n_samples = self.hparams.n_samples if n_samples is None else n_samples
        # 确保 prompt 的维度是 (n_samples, prompt_len, model_del)
        prompt["lm_input"] = prompt["lm_input"].unsqueeze(0).expand(n_samples, -1, -1)
        # 定义奖励函数，预先指定一些参数
        reward_fn = partial(
            self.reward.score,                  # score 为奖励函数本体
            prompt_length=prompt["lm_input"].shape[1],      # prompt_len 为 prompt 的长度
            model=self.model,                   # 传入 model
        )
        # 调用 generate_and_return_termination_logprob 函数生成文本并计算相关概率和奖励。
        (
            generated_text,         # 生成的文本, [B, T], 不包含 prompt
            lm_input_embedding,     # prompt embedding 加上生成文本的 embedding
            log_pf,                 # 前向概率的对数
            log_pterm,              # 句子终止概率的对数
            log_r,                  # 奖励的对数
            log_r_unpenalized,      # 未惩罚奖励的对数
        ) = generate_and_return_termination_logprob_repetition_penalty(
            self.model,                                                 # 传入 model        
            prompt,                                                     # 传入 prompt
            reward_fn=reward_fn,                                        # 传入奖励函数
            termination_token_id=self.end_of_sentence_token_id,         # 句子结束符的 token id
            vocab_naughty_mask=self.hparams.illegal_token_mask,
            min_len=self.hparams.min_sentence_len,                      # 最小句子长度
            max_len=self.hparams.max_sentence_len,                      # 最大句子长度
            temperature=pf_temperature,                                 # 采样温度
            skip_rewards=False,                                         # 不计算奖励
            action_seq=action_seq,                                      # 指定策略
        )
        return generated_text, lm_input_embedding, log_pf, log_pterm, log_r, log_r_unpenalized

    """
    定义单个训练步的操作，计算损失并返回字典形式的结果。用于训练模型
    prompt：提示
    batch_idx：当前批次的索引
    """
    def training_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        # print(prompt)
        assert isinstance(prompt, dict)
        prompt["lm_input"] = prompt["lm_input"][0]  # [T, embedding_d]

        # 决定是否从奖励缓冲区中进行采样，这里概率为0.25（设置文件中有），并且此时缓冲区有存对应提示
        # Sample a sentence and get the reward
        if (
            random.random() < self.hparams.use_buffer_prob
            and self.reward_buffer.sample(self.hparams.n_samples, prompt["lm_input"])[0] is not None    # 保证缓冲区中有可用样本
        ):
            # 从奖励缓冲区中采样，得到 action_seq 和 reward（log_r）
            # Using a sample from the reward buffer
            action_seq, log_r = self.reward_buffer.sample(
                self.hparams.n_samples, prompt["lm_input"]
            )
            # 生成句子并计算前向概率、句子终止概率、未惩罚奖励
            generated_text, lm_input_embedding, log_pf, log_pterm, _, log_r_unpenalized = self.forward(
                prompt, action_seq=action_seq
            )
            log_r = log_r[
                :, : generated_text.shape[1]
            ]  # Undo padding from buffer   # 去除生成文本中多余的填充部分（填充是0）
            log_r *= 1 / self.reward.temperature  # redo the effect of reward tempering
        else:
            # 不用缓冲区就直接用 forward 采样
            # Using the forward policy
            # 概率进行温度调整
            if random.random() < self.hparams.pf_temp_prob:  # With tempering
                # 在设定的下限和上限之间随机取一个值（0.5~2.0）
                pf_temp = (
                    random.random()
                    * (self.hparams.pf_temp_high - self.hparams.pf_temp_low)
                    + self.hparams.pf_temp_low
                )
            # 不进行温度调整
            else:  # Without tempering
                pf_temp = 1.0
            # 从 forward 中进行完整采样，并进行温度调整
            generated_text, lm_input_embedding, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, pf_temperature=pf_temp
            )
            # 将采样到的句子添加到奖励缓冲区
            self.reward_buffer.add_batch(
                prompt=prompt["lm_input"],
                sentences=generated_text,
                logrewards=log_r
                * self.reward.temperature,  # undo the effect of reward tempering
            )
        
        # 计算 modified_subtb_loss（子轨迹平衡损失），也是 GFN 的损失
        # Get the GFN loss
        loss = trajectory_balance_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt["lm_input"]),
            subtb_lambda=self.hparams.subtb_lambda,
        )
        # print(f"loss: {loss}")
        # 进行log记录和计算
        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt["lm_input"]),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train/logR",
            last_log_r.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (avg)",
            log_ps.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (max)",
            log_ps.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/sentence_len",
            sentence_len.float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    """
    定义单个验证步的操作，计算损失并返回字典形式的结果。用于训练模型
    prompt：提示
    batch_idx：当前批次的索引
    """
    def validation_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        prompt["lm_input"] = prompt["lm_input"][0]

        # Sample a sentence and get the reward
        generated_text, lm_input_embedding, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
            prompt
        )

        # Get the GFN loss
        loss = trajectory_balance_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt["lm_input"]),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        if torch.isinf(loss.clone().detach()):
            loss = 0.0
            with open("loss_is_inf.txt", 'a') as f:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] log_r: {log_r}\n")
                f.write(f"[{timestamp}] log_pf: {log_pf}\n")
                f.write(f"[{timestamp}] log_pterm: {log_pterm}\n")
                f.write(f"[{timestamp}] generated_text: {generated_text}\n")
                f.write(f"[{timestamp}] subtb_lambda: {self.hparams.subtb_lambda}\n")
                f.write("\n")

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt["lm_input"]),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/logR",
            last_log_r.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (avg)",
            log_ps.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (max)",
            log_ps.max(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            sync_dist=True,
        )
        self.log(
            "val/sentence_len",
            sentence_len.float().mean(),
            sync_dist=True,
        )
        # NOTE: 这里暂时注释掉了，为了运行
        # if self.diversity_metric.method is not None:
        #     generated_sentences = self.tokenizer.batch_decode(
        #         generated_text[:, len(prompt) :]
        #     )
        #     generated_sentences = [
        #         text.replace(".", "") for text in generated_sentences
        #     ]
        #     diversity = self.diversity_metric(generated_sentences)
        #     self.log(f"val/{self.diversity_metric_name}", diversity, sync_dist=True)

    """
    定义每个训练步开始时的操作，包括更新奖励温度和学习率。
    prompt：提示
    batch_idx：当前批次的索引
    """
    def on_train_batch_start(self, prompt, batch_idx):
        # Update scheduled quantities
        # 根据全局步数更新奖励温度和学习率
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        lr = self.get_lr_at_step(self.global_step)
        self.reward.temperature = reward_temp
        for pg in self.optimizers().param_groups:
            pg["lr"] = lr

    """
    定义每个训练轮次的开始时的操作，包括记录当前步的奖励温度和学习率，并进行探针采样。
    """
    def on_train_epoch_start(self):
        # 记录奖励温度和学习率，并确保在分布式训练环境中同步日志
        # Log scheduled quantities
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        self.log("scheduled/lr", self.get_lr_at_step(self.global_step), sync_dist=True)
        
        with open("loss_is_inf.txt", 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] global_step: {self.global_step}\n")

        # 记录探针样本，存储至探针表中
        # NOTE: 探针没有适配
        # Log probe samples
        if (
            self.hparams.train_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.train_probes)
            self.logger.log_table("samples/train_probes", dataframe=samples_table)

    """
    定义每个验证轮次的开始时的操作
    """
    def on_validation_epoch_start(self):
        # Log variance of (logR - logP(s)) using exploration, which should be 0.0
        log_rs, log_pfss = [], []
        val_data = self.trainer.datamodule.val_dataloader().dataset
        for prompt in val_data:
            prompt["lm_input"] = prompt["lm_input"][0].to(self.device)
            # 生成文本，并计算相关的概率和奖励，2.0 表示使用较高的温度进行探索
            generated_text, lm_input_embedding, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, pf_temperature=2.0
            )
            # 调用方法计算终止位置的 log_pfs（各个生成的语句的累计前向概率加上位置终止的概率）
            # log_r（各个生成语句的直接奖励分数） 等值进行聚合并作记录
            log_pfs, log_r, _, _ = get_termination_vals(
                generated_text=generated_text,
                log_pf=log_pf,
                log_pterm=log_pterm,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(prompt["lm_input"]),
            )
            log_rs.append(log_r)
            log_pfss.append(log_pfs)
        log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        # Log probe samples
        if (
            self.hparams.val_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.val_probes)
            self.logger.log_table("samples/val_probes", dataframe=samples_table)

    """
    在训练开始时执行的函数，用于初始化，并先行使用 baseline 方法进行采样
    """
    def on_train_start(self):
        # Log baseline metrics
        val_data = None
        # NOTE: baseline 方法没有适配
        # baseline_performance = None
        # for prompt in val_data:
        #     prompt = prompt[0]
        #     # 使用 baseline 方法进行采样
        #     samples = self.sample_baselines(
        #         prompt.to(self.device), n_samples=self.hparams.n_samples
        #     )
        #     if baseline_performance is None:
        #         baseline_performance = pd.DataFrame(
        #             data=np.zeros((6, len(samples))),
        #             columns=samples.keys(),
        #             index=[
        #                 "logP(s) (avg)",
        #                 "logP(s) (max)",
        #                 "logP(s) unpenalized (avg)",
        #                 "logP(s) unpenalized (max)",
        #                 self.diversity_metric_name,
        #                 "sentence length",
        #             ],
        #         )
        #     for baseline in samples:
        #         baseline_performance.loc["logP(s) (avg)", baseline] += samples[
        #             baseline
        #         ]["logP(s)"].mean().item() / len(val_data)
        #         baseline_performance.loc["logP(s) (max)", baseline] += samples[
        #             baseline
        #         ]["logP(s)"].max().item() / len(val_data)
        #         baseline_performance.loc[
        #             "logP(s) unpenalized (avg)", baseline
        #         ] += samples[baseline]["logP(s) unpenalized"].mean().item() / len(
        #             val_data
        #         )
        #         baseline_performance.loc[
        #             "logP(s) unpenalized (max)", baseline
        #         ] += samples[baseline]["logP(s) unpenalized"].max().item() / len(
        #             val_data
        #         )
        #         if samples[baseline][self.diversity_metric_name] is None:
        #             baseline_performance.loc[
        #                 self.diversity_metric_name, baseline
        #             ] = None
        #         else:
        #             baseline_performance.loc[
        #                 self.diversity_metric_name, baseline
        #             ] += samples[baseline][self.diversity_metric_name] / len(val_data)
        #         baseline_performance.loc["sentence length", baseline] += samples[
        #             baseline
        #         ]["sentence length"].float().mean().item() / len(val_data)
        # baseline_performance = baseline_performance.reset_index(names="metric")
        # if self.logger is not None:
        #     self.logger.log_table(
        #         "val/baseline performance", dataframe=baseline_performance
        #     )

        # # Log baseline probes
        # if self.hparams.val_probes is not None and self.logger is not None:
        #     samples_table = self.sample_probes_baselines(self.hparams.val_probes)
        #     self.logger.log_table(
        #         "samples/val_probes (baselines)", dataframe=samples_table
        #     )

    """
    自定义方法，用于从探针数据中采样样本，
    """
    def sample_probes(self, probes, n_samples=4):
        # 确保 probes 是一个列表，并且每个元素都是一个一维张量，其中存储的是待推理的样本
        assert isinstance(probes, list) and probes[0].ndim == 1
        # 存储生成的样本
        samples = []
        for probe in probes:
            # 取出待推理的文本
            probe_str = self.tokenizer.decode(probe)
            # 进行推理
            with torch.no_grad():
                generated_text, _, _, log_r, log_r_unpenalized = self.forward(
                    probe.to(self.device), n_samples=n_samples
                )
            # 计算终止位置的相关值，包括奖励分数log_r、未惩罚奖励分数log_r_unpenalized。
            log_ps, log_ps_unpenalized = get_termination_vals(
                generated_text=generated_text,
                log_pf=None,
                log_pterm=None,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(probe),
            )[1:3]
            # undo temperature
            log_ps *= self.reward.temperature
            log_ps_unpenalized *= self.reward.temperature

            # 生成中间的文本
            generated_text = generated_text[:, len(probe) :]
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]
            # 进行 log 记录
            for i in range(len(generated_text)):
                samples.append(
                    {
                        "Prompt": probe_str,
                        "Sampled sentence": generated_text[i],
                        "logP(s)": log_ps[i].item(),
                        "logP(s) unpenalized": log_ps_unpenalized[i].item(),
                    }
                )
        # 返回探针采样结果
        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt", "logP(s)"], ascending=False)
        return samples

    """
    自定义方法，用于使用基线方法进行探针采样。
    """
    def sample_probes_baselines(self, probes, n_samples=4):
        # probes 为 prompt 列表，并确保每个元素都是一个一维张量
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            # 解码为文本
            probe_str = self.tokenizer.decode(probe)
            # 使用基线方法进行采样
            probe_samples = self.sample_baselines(
                probe.to(self.device), n_samples=n_samples
            )
            for i in range(n_samples):
                sample = {"Prompt": probe_str}
                for baseline in probe_samples:
                    sample[f"Sampled sentence ({baseline})"] = probe_samples[baseline][
                        "sample"
                    ][i]
                    sample[f"logP(s) ({baseline})"] = probe_samples[baseline][
                        "logP(s)"
                    ][i].item()
                    sample[f"logP(s) unpenalized ({baseline})"] = probe_samples[
                        baseline
                    ]["logP(s) unpenalized"][i].item()
                samples.append(sample)

        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt"], ascending=False)

        return samples

    """
    自定义方法，用于从提示数据中采样基线样本。
    """
    def sample_baselines(self, prompt, n_samples=4):
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
        
        # 确保 prompt 是一个一维张量，并进行维度扩张方便批处理
        assert prompt.ndim == 1
        prompt = prompt.unsqueeze(0)

        def generate(prompt, **kwargs):
            with torch.no_grad():
                # 切换为 base 模型
                lora_to_base(self.model)
                # 使用 generate 方法生成文本
                generated_text = self.model.generate(
                    prompt,
                    min_new_tokens=self.hparams.min_sentence_len,
                    max_new_tokens=self.hparams.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=torch.from_numpy(self.hparams.illegal_token_mask)
                    .nonzero()
                    .squeeze(-1),
                    **kwargs,
                )
                # 切换回微调模型
                base_to_lora(self.model)

                # 计算语句得分
                log_r, log_r_unpenalized = self.reward.score(
                    generated_text,
                    prompt_length=prompt.shape[1],
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                (   # 计算终止相关值
                    _,
                    last_log_r,
                    last_log_r_unpenalized,
                    sentence_len,
                ) = get_termination_vals(
                    generated_text=generated_text,
                    log_pf=None,
                    log_pterm=None,
                    log_r=log_r,
                    log_r_unpenalized=log_r_unpenalized,
                    termination_token_id=self.end_of_sentence_token_id,
                    prompt_len=prompt.shape[1],
                )
                # 恢复温度（undo temperature）
                log_ps = last_log_r * self.reward.temperature
                log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

            # 去除提示部分，保留生成的文本
            generated_text = generated_text[:, prompt.shape[1] :]
            # 将填充的 token id 更换为结束 token id
            generated_text = torch.where(
                generated_text == self.tokenizer.eos_token_id,
                self.end_of_sentence_token_id,
                generated_text,
            )
            # 转为字符串文本
            generated_text = self.tokenizer.batch_decode(generated_text)
            # 移除文本中的句号
            generated_text = [text.replace(".", "") for text in generated_text]
            
            # 如果生成了多个样本，则计算多样性指标
            if len(generated_text) > 1:
                diversity = self.diversity_metric(generated_text)
            else:
                diversity = None

            # 如果只生成了一个样本，则将其复制 n_samples 次，并扩展相应的奖励分数和未惩罚的奖励分数
            if len(generated_text) == 1:
                generated_text = generated_text * n_samples
                log_ps = log_ps.expand(n_samples, -1)
                log_ps_unpenalized = log_ps_unpenalized.expand(n_samples, -1)

            return {
                "sample": generated_text,
                "logP(s)": log_ps,
                "logP(s) unpenalized": log_ps_unpenalized,
                "sentence length": sentence_len,
                self.diversity_metric_name: diversity,
            }

        # 这里使用了束搜索、多样束搜索、核搜索、语言模型采样、带温度的语言模型采样、贪心采样
        # 带 [fair] 表示变体，一般是使用了较小的束宽
        samples = {}

        # Beam search
        samples["beam"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples * 5,
            length_penalty=0.0,
        )
        samples["beam [fair]"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples,
            length_penalty=0.0,
        )

        # Diverse beam search
        samples["diverse beam"] = generate(
            prompt=prompt,
            num_beams=n_samples * 5,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )
        samples["diverse beam [fair]"] = generate(
            prompt=prompt,
            num_beams=n_samples,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )

        # Nucleaus sampling
        samples["nucleus"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            top_p=0.95,
        )

        # LM
        samples["LM"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
        )

        # LM with temperature
        samples["LM tempered"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            temperature=self.hparams.reward_temp_end,
        )

        # Greedy
        samples["greedy"] = generate(
            prompt=prompt,
            do_sample=False,
        )

        return samples

    """
    定义优化器和学习率调度器，返回优化器对象或优化器配置字典
    """
    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
