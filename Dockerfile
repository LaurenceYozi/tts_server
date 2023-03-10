FROM python:3.8-slim-bullseye
ENV LANG C.UTF-8
RUN apt-get update && apt-get -y install ffmpeg gnupg2 git wget vim curl locales python3 python3-pip python3-dev 

# Fix locale error
RUN apt-get install -y locales
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN locale-gen
ENV LC_CTYPE en_US.UTF-8

WORKDIR /home/ubuntu/TTS/English
# Get project source code
COPY . .

# Build project
RUN pip3 install --upgrade pip setuptools
RUN pip3 install numpy==1.18.5
#RUN pip3 install -r male_requirements.txt
#RUN pip3 uninstall -y tensorflow-gpu
RUN pip3 install -r requirements.txt
#RUN pip3 install -r nemo_requirements.txt
#RUN bash jemalloc.sh

# Get NLTK tagger 
RUN python3 get_nltk_tagger.py

EXPOSE 8508
CMD [ "bash", "script.sh"]