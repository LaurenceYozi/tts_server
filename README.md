# Text to speech using python and streamlit
- streamlit 需要 
    - requirement.txt 安裝 pip 軟件
    - packages.txt 安裝 linux 軟件

# Usage
```bash
#自行建立新環境
conda create -n {name} python=3.8    # python > 3.6
pip install numpy==1.18.5
pip install -r requirements.txt
pip install -r nemo_requirements.txt # 僅在 linux 可使用
bash jemalloc.sh    # 內存管理器，可裝可不裝
python3 get_nltk_tagger.py
```

## Docker build
```bash
(step.1)建立docker image file
# 重新包裝docker images
sudo docker build --no-cache -t "tts_eng" -f Dockerfile .

(step.2) 透過新的image檔案，跑起container服務
# 開啟新的服務 # 可以到到action
sudo docker run --name tts_eng_web -p 8288:8508 tts_eng:latest
# 開啟新的服務(背景執行)
sudo docker run --name tts_eng_web -d -p 8288:8508 tts_eng:latest
```