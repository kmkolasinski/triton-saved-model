# Triton Python Backend for Multimodel/Signatures Inference Demo

* This repository contains code for running multiple models on Triton Inference Server with python backend
* The python backend code can handle multiple models and multiple signatures
* This project was prepared for educational purposes, to show how we can use Triton Inference Server with python backend to simulate similar API as TFServing

```bash
docker compose build
```

## Running notebooks

* Notebooks were tested with python version 3.11.4, see [requirements.txt](requirements.txt)
* Use [export-classifier.ipynb](notebooks%2Fexport-classifier.ipynb) to export various classifiers. 
* Triton and TFServing will be reading these models from [models.conf](data%2Fmodels.conf) configuration file.
* Then use [run-client.ipynb](notebooks%2Frun-client.ipynb) to run the client and benchmark the performance
* See [docker-compose.yml](docker-compose.yml) for the available services

## Running servers

* Firstly, use [export-classifier.ipynb](notebooks%2Fexport-classifier.ipynb) to export various classifiers
* To start triton server, run the following command
```bash
docker compose up triton_server 
```
* To start the Tensorflow Serving server, run the following command
```bash
docker compose up tf_serving_server
```

## Benchmark results

* Two types of architectures were tested and exported to the classifier Modules used by servers:
  * ResNet50
    * standard SavedModel
    * SavedModel compiled with XLA and AMP
  * EfficientNetB0
    * standard SavedModel
    * SavedModel compiled with XLA and AMP

* The benchmarks were performed on NVIDIA RTX A4000 GPU with 8GB of memory
* Each benchmark was run for 500 iterations to predict batch of 100 images of size 224x224 (50k images in total)
* I benchmarked only the `images` signature, which accepts images tensor of shape [batch, 224, 224, 3]
* When running the models locally with TF python API, I got following results:

| Model          | Architecture       | Time [s] |
|----------------|--------------------|----------|
| ResNet50       | SavedModel         | 57       |
| ResNet50       | SavedModel XLA/AMP | 25       |
| EfficientNetB0 | SavedModel         | 52       |
| EfficientNetB0 | SavedModel XLA/AMP | 13       |

* Running same benchmark but with Triton Inference Server (4 client threads, 1 server instance), I got following results:

| Model          | Architecture       | Time [s] |
|----------------|--------------------|----------|
| ResNet50       | SavedModel         | 73       |
| ResNet50       | SavedModel XLA/AMP | 23       |
| EfficientNetB0 | SavedModel         | 54       |
| EfficientNetB0 | SavedModel XLA/AMP | 17       |

* Running same benchmark but with Triton Inference Server (4 client threads, **2 server instances**), I got following results:

| Model          | Architecture       | Time [s] |
|----------------|--------------------|----------|
| ResNet50       | SavedModel         | 76       |
| ResNet50       | SavedModel XLA/AMP | 20       |
| EfficientNetB0 | SavedModel         | 54       |
| EfficientNetB0 | SavedModel XLA/AMP | 13       |

* For Tensorflow Serving I was not able to text XLA/AMP models, I got following error when trying to serve them:
```bash
UNIMPLEMENTED: Could not find compiler for platform CUDA: NOT_FOUND: could not find registered compiler for platform CUDA
```
* The results for TF Serving were as follows (excluding XLA/AMP models):

| Model          | Architecture       | Time [s]              |
|----------------|--------------------|-----------------------|
| ResNet50       | SavedModel         | 59                    |
| ResNet50       | SavedModel XLA/AMP | CUDA: NOT_FOUND error |
| EfficientNetB0 | SavedModel         | 51                    |
| EfficientNetB0 | SavedModel XLA/AMP | CUDA: NOT_FOUND error |

Also, I noticed that when using TFServing, GPU memory was higher than when using Triton Inference Server. 
I was getting `OOM when allocating tensor with shape[100,56,56,256]`  when using num_workers=10