
## Docker


#### Dataset
Build dataset creator with the following command: (this is quick)
> ```docker build -t dataset -f dataset.Dockerfile .```

Run dataset collector with the following command: (this is slow)
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset dataset`

#### Training

Build the training image with the following command where `CUDA_VERSION` can be one of `cu116, cu113, cu102, cpu`: (this is quite slow)
> `docker build -t token_completion . --build-arg CUDA_VERSION=[CUDA_VERSION]`

Run the trainer with the following command where `MAKE_TARGET` is a target from [Makefile](Makefile): 
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset token_completion [MAKE_TARGET]`