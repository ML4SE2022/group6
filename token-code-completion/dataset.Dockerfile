FROM python:3.7

RUN pip --no-cache install jsbeautifier datasets tqdm

WORKDIR /dataset/javaCorpus
COPY dataset/javaCorpus/download.sh .
COPY dataset/javaCorpus/preprocess.py .

WORKDIR /dataset/py150/
COPY dataset/py150/download_and_extract.sh .
COPY dataset/py150/preprocess.py .

WORKDIR /dataset/javascriptAxolotl/
COPY dataset/javascriptAxolotl/download.py .

WORKDIR /dataset/typescriptAxolotl/
COPY dataset/typescriptAxolotl/download.py .

WORKDIR /
COPY entrypoint.sh .
CMD ["bash", "entrypoint.sh"]
# JavaCorpus
#RUN cd javaCorpus && ./download.sh && python preprocess.py

#RUN cd /dataset/py150 && ./download_and_extract.sh