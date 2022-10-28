
FROM python:3.7 as base
#FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
         && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
         && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
               sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
               tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y libnvidia-container1 libnvidia-container-tools \
    && rm -rf /var/lib/apt/lists/*

#apt-get install nvidia-container-runtime


RUN pip --no-cache install poetry
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry config virtualenvs.create false && poetry --no-cache install

ARG CUDA_VERSION

RUN pip --no-cache install "torch==1.12.0+${CUDA_VERSION}" "torchvision==0.13.0+${CUDA_VERSION}" torchaudio==0.12.0 --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

RUN pip --no-cache install transformers

COPY code/ code/

COPY run-all.sh eval-all.sh /

COPY Makefile .

ENTRYPOINT ["make"]
CMD ["all"]