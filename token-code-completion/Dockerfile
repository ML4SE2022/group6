FROM python:3.7 as base

RUN pip --no-cache install poetry
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry --no-cache install

ARG CUDA_VERSION

RUN pip --no-cache install "torch==1.12.0+${CUDA_VERSION}" "torchvision==0.13.0+${CUDA_VERSION}" torchaudio==0.12.0 --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

ENTRYPOINT ["make"]
CMD ["all"]