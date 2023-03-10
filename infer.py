from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import yaml
import os
import re
from model.models import ForwardTransformer
from data.audio import Audio
import torch
#from nemo_text_processing.text_normalization.normalize import Normalizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from g2p_en import G2p
g2p = G2p()
_punctuation = "-!'(),.:;? "


class bcolors:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    reset = '\033[0m' #RESET COLOR


# file path settings
fs2_config_path = "./fs2_model/config.yaml"
fs2_female_model_path = "./ckpt/step_75000_v20_147/"
fs2_male_model_path = "./fs2_model/model_male.h5"
parallel_waveGAN_config_path = "./parallel_waveGAN/config.yaml"
parallel_waveGAN_male_model_path = "./parallel_waveGAN/generator_male.h5"

# initialize fastspeech2 model.
print(f"Loading FastSpeech2...\n{bcolors.red}\t {fs2_female_model_path}{bcolors.reset}")
FFT_model = ForwardTransformer.load_model(fs2_female_model_path)
audio = Audio.from_config(FFT_model.config)

print("Loading MelGAN...")
vocoder = torch.hub.load('./melgan', 'melgan', source='local')
vocoder.eval()


# initialize normalizer
print("Initialize Normalizer...")
#normalizer = Normalizer(input_case='cased', lang='en')


def fs2_infer(gender, phone_seq="", input_ids=None, speed=1.0):
    if input_ids is None:
        input_ids = []
    # fastspeech inference
    if gender == "female":
        mel = FFT_model.predict(phone_seq, speed_regulator = speed)
    return mel


def melgan_infer(input_mel):    
    with torch.no_grad():
        audio = torch.tensor(input_mel['mel'].numpy().T[np.newaxis,:,:])    # shape (1, 80, mel_length)
        #audio = torch.tensor(input_mel)    # test vocoders with true mel spectrum (lines 162, 167)
        return vocoder.inference(audio)


def punctuation_handling(text):
    count = 0
    remove_repeat_text = ""
    p = [",", ".", ";", ":", "!", "?", "(", ")", "-", "'"]
    space_p = [" ", ",", ".", ";", ":", "!", "?", "(", ")", "-", "'"]
    if len(text) == len(np.where(np.array(list(text)) == ".")[0]):
        for _ in range(len(text)):
            text += " dot"
            print(text)
    elif len(text) == len(np.where(np.array(list(text)) == ",")[0]):
        for _ in range(len(text)):
            text += " comma"

    while text[count] in space_p:    # avoid punctuation at the beginning
        count += 1
    print(bcolors.green + "\nRemove excess amount of punctuation at the beginning：" + bcolors.reset, count)
    text = text[count:]
    for i in range(len(text)):
        if text[i] in p:
            if i != len(text) - 1 and text[i + 1] in p:
                continue
            elif i == len(text) - 1:
                remove_repeat_text += "." if text[i] in [",", ";"] else text[i]
            elif text[i] == ";":
                remove_repeat_text += ","
            elif text[i] == "'":
                if text[i - 1] == " ":
                    continue
                elif text[i + 1] == " ":
                    remove_repeat_text += ","
                else:
                    remove_repeat_text += text[i]
            else:
                remove_repeat_text += text[i]
        elif text[i] == " " and text[i + 1] == "'":
            remove_repeat_text += ", " if text[i-1] not in p else " "
        else:
            remove_repeat_text += text[i]
    return remove_repeat_text


def nsw_idx(text, non_standard_words):
    index_dict = {}
    counter = 0
    for i in non_standard_words:
        index = -1
        # 最多偵測三次位置，可依照需求修改
        for _ in range(3):
            index = text.find(i, index + 1)
            # find 沒找到字串，回傳 -1，找到則回傳第一個字的 index
            if index != -1:
                index_dict[f"{i}_{str(index)}"] = index    # [('UK_32', 32), ('it engineer_44', 44)....]
                counter += 1
            else:
                break
    return counter, index_dict


def fix_non_standard_words(text, index):
    count = 0
    for k, v in index:
        check = False
        length = len(re.split("_|\s", k)[0])
        if v != -1:
            v += count
            temp = ""
            for i in range(len(text)):
                if i == v:
                    # 將非標準詞加上空字串分隔, ex: UK → U K
                    temp += f"{text[i]} "
                    length -= 1
                    check = True
                    # 經過1個非標準詞會新增一個空白，因此後方的非標準詞 index + 1
                    count += 1
                elif check == True and length != 1:
                    # ex: UK's → U K's
                    if text[i + 1] == "'" and text[i + 2] == "s":
                        temp += text[i]
                        length -= 2
                    else:
                        temp += f"{text[i]} "
                        length -= 1
                        count += 1
                else:
                    temp += text[i]
            text = temp
    #print("fix:", text)
    return text


def infer(text, gender="female", speed=1.0):
    # remove sentence beginnings and repeated punctuation
    text = punctuation_handling(text)

    # Nemo text normalizatopm tool
    #text = normalizer.normalize(text)
    # 若有新的 (NeMo 無法處理) 非標準詞，請加入以下列表，將以空白及 '_' 分割抓出最前方非標準詞
    non_standard_words = ["AM", "UK", "CV", "CCTV", "IT engineer", "PE", "UN"]
    counter, index_dict = nsw_idx(text, non_standard_words)
    if counter > 0:
        index_dict = sorted(index_dict.items(), key=lambda item:item[1])
        text = fix_non_standard_words(text, index_dict)
        print(bcolors.green + "**** Additional non-standard words index ****\n" + bcolors.reset, index_dict)
    if text[len(text) - 1] not in [".", "!", "?", "(", ")"]:
        text += '.'

    # 移除對話句子 A、B 角色 A:Which skirt do you prefer? B:This skirt can show my long legs.
    remove_AB_text = ""
    remove_AB_count = 0
    for i in range(len(text)):
        if remove_AB_count > 0:
            remove_AB_count -= 1
            continue
        if i != (len(text) - 1) and text[i] in ["A", "B", "C"] and text[i + 1] == ":":
            remove_AB_count = 1
            continue
        elif text[i] == ":":
            remove_AB_text += "," 
        else:
            remove_AB_text += text[i]
    text = remove_AB_text
    print(bcolors.green + "\n**** Text Normalization ****\n" + bcolors.reset, text, "\n")

    # word to phoneme
    phone_seq, converted_text = g2p(text)
    phone_seq = ' '.join(phone_seq).split("   ")
    phone_seq = ' '.join(list(map(lambda x: f"{{{x}}}" if x not in _punctuation else x, phone_seq)))

    # synthesis
    if gender == "female":
        mel_outputs = fs2_infer(gender, phone_seq=phone_seq, speed=speed)
        #mel_outputs = fs2_infer(gender, phone_seq=phone_seq, speed=speed)  # 模型預測已將 speed 做倒數
        #mel_outputs = torch.load("test_true_melspec.mel").numpy()    # test vocoders with true mel spectrum

        # inference
        audio = melgan_infer(mel_outputs) 
        #audio = hifigan_infer('hifigan/mels/mel.npy') 
        mel_length = mel_outputs['mel'].shape[0]
        #mel_length = mel_outputs.shape[2]    # test vocoders with true mel spectrum
    # save to file
    randint = np.random.randint(9999999)
    wav_path = f"temp/{str(randint)}.wav"
    sf.write(wav_path, audio, 22050, "PCM_16")
    print(f"{bcolors.red}**** Output file: {wav_path} ****\n{bcolors.reset}")
    return wav_path, converted_text, phone_seq, mel_length


# def visualize_mel_spectrogram(mels):
#     #matplotlib inline
#     mels = tf.reshape(mels, [-1, 80]).numpy()
#     fig = plt.figure(figsize=(10, 8))
#     ax1 = fig.add_subplot(311)
#     ax1.set_title('Predicted Mel-Spectrogram')
#     im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
#     fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
#     plt.show()
#     plt.close()
