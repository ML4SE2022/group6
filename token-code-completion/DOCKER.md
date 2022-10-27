
## Docker


#### Dataset
Build dataset creator with the following command: (this is quick)
> ```docker build -t dataset -f dataset.Dockerfile .```

Run dataset collector with the following command: (this is slow)
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset dataset`

#### Training

Build the training image with the following command where `CUDA_VERSION` can be one of `cu116, cu113, cu102, cpu`: (this is quite slow)
> `docker build -t token_completion . --build_arg CUDA_VERSION=cu116`

Run the trainer with the following command:
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset token_completion`