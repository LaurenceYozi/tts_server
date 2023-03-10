import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import streamlit as st
import os
import time
import glob
import os


from gtts import gTTS
from googletrans import Translator
from infer import infer

try:
    os.mkdir("temp")
except:
    pass
st.title("Text to speech")
translator = Translator()

text = st.text_input("Enter text")
in_lang = st.selectbox(
    "Select your input language",
    ("English", "Hindi", "Bengali", "korean", "Chinese", "Japanese"),
)
if in_lang == "English":
    input_language = "en"
elif in_lang == "Hindi":
    input_language = "hi"
elif in_lang == "Bengali":
    input_language = "bn"
elif in_lang == "korean":
    input_language = "ko"
elif in_lang == "Chinese":
    input_language = "zh-cn"
elif in_lang == "Japanese":
    input_language = "ja"

st.write("### **Output language: English**")
output_language = "en"

# out_lang = st.selectbox(
#     "Select your output language",
#     ("English", "Chinese"),
# )
# if out_lang == "English":
#     output_language = "en"
# elif out_lang == "Chinese":
#     output_language = "zh-cn"

speed = st.slider(label='Speech rate', min_value=0.5, max_value=1.0, value=0.8, step=0.05)


def text_to_speech(input_language, output_language, text, tld, speed):
    translation = translator.translate(text, src=input_language, dest=output_language)
    trans_text = translation.text
    #tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
    wav_path, _, _, _ = infer(trans_text, gender="female", speed=speed)
    # try:
    #     my_file_name = text[0:20]
    # except:
    #     my_file_name = "audio"
    # tts.save(f"temp/{my_file_name}.mp3")
    return wav_path, trans_text


display_output_text = st.checkbox("Display output text")

if st.button("convert"):
    result, output_text = text_to_speech(input_language, output_language, text, tld="com", speed=speed)
    audio_file = open(result, "rb")
    audio_bytes = audio_file.read()
    st.markdown("## Your audio:")
    st.audio(audio_bytes, format="audio/wav", start_time=0)

    if display_output_text:
        st.markdown("## Output text:")
        st.write(f" {output_text}")


def remove_files(n):
    mp3_files = glob.glob("temp/*wav")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)


remove_files(7)
