import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('output_trump3.wav', 16000)    # 英文
# prompt_speech_16k = load_wav('xlpj.wav', 16000)             # 中文

# ----------------- 测试用，英文合成数据集 ----------------- # 
tts_list = []
with open("inference_target_tts_text.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        tts_list.append(line.strip())
# ----------------- 测试用，中文合成数据集 ----------------- # 
# tts_list = ["新的一年里，祝大家工作顺利，学业有成，身体健康，大头、小头、兔头三头齐发展，也欢迎大家往史群搬运更多的史。最后，再次祝大家新春快乐！",
#             "中文我也尝试了，在中文上也有这个趋势，会稍微快一点。",
#             "这个和上次训练的结果正好相反，上一个是长度往上飙升，这个是长度往下飙升。",
#             "现在正在跑这个哈哈哈，我感觉训练过程中间的可能会生成好的，所以也想试一试。",
#             "这会不会是本来模型确实在努力收敛，但是最后还是找到了捷径的情况呀。",
#             "张老师，值此新春佳节之际，陈林在这里衷心祝你新春快乐！愿你新的一年心情愉悦，身体健康，万事顺遂！",
#             "这里面还包括了一些类似呼吸声音这种。",
#             "声音有一点不稳定，不过还可以，可能是因为温度才开始下降，模型还没拟合。"]


# 现在这个可以合成 list 目标
for i, j in enumerate(cosyvoice.inference_zero_shot(tts_list[:9],
                                                    #  'Because our leaders are stupid, our politicians are stupid',
                                                       '各位观众朋友们大家好，我是从来不带节奏的血狼，今天给大家做一期明日方舟终末地基建入门教程。',
                                                       prompt_speech_16k, stream=False)):
    torchaudio.save('inference_results/test/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)