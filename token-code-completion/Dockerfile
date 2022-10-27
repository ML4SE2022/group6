FROM python:3.7 as base

RUN pip --no-cache install poetry
COPY pyproject.toml .
RUN poetry --no-cache install

ARG cuda_version=cu116

RUN pip --no-cache install "torch==1.12.0+${cuda_version}" "torchvision==0.13.0+${cuda_version}" torchaudio==0.12.0 --extra-index-url "https://download.pytorch.org/whl/${cuda_version}"

ENTRYPOINT ["make"]
CMD ["all"]