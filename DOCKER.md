
## Docker

## Prerequisites (on the host device)

- Install NVidia drivers from [here](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- Install docker
- Make sure `nvidia-container-toolkit` is installed
- Any nvidia driver issues are left as an exersise to the reader
- 

#### Dataset
Build dataset creator with the following command: (this is quick)
> ```docker build -t dataset -f dataset.Dockerfile .```

Run dataset collector with the following command: (this is slow)
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset dataset`

#### Training

Build the training image with the following command where `CUDA_VERSION` can be one of `cu116, cu113, cu102, cpu`: (this is quite slow)
> `docker build -t token_completion . --build-arg CUDA_VERSION=[CUDA_VERSION]`

Run the trainer with the following command where `MAKE_TARGET` is a target from [Makefile](Makefile): 
> `docker run --gpus all --mount type=bind,source=$(pwd)/dataset,target=/dataset --mount type=bind,source=$(pwd)/logs,target=/logs token_completion [MAKE_TARGET]`
