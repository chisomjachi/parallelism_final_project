# 6998Final-Project

## Project:

Our project focus on Parallelism and it's influence on scaling efficiency and throughput. 
We create different experiments with Model Parallelism and Data Parallelism to show how
execution time and convergence time change for a single model changes with different set
up of parallelism. 

## Repository:

Model Parallelism: 

In the python notebook FINAL_PROJECT_6998.ipynb I process CIFAR-10 data
and use the ResNet50 as main model. I also create a smaller randomly generated data set to
compare it's performance with CIFAR-10. Then I run Model Parallelism and Pipepline 
Parallelism experiments on both dataset and produce comparison graphs on execution time of 
training of the model. 

## Commands:

Model Parallelism: Simply run python notebook. 

## Results:

Model Parallelism & Single GPU on Small Dataset 

![Unknown-5](https://user-images.githubusercontent.com/44592326/208521935-9fa1b291-7dcb-4f87-87dc-45dc37f58d3a.png)

Model Parallelism & Single GPU on Small Dataset on CIFAR-10

![Unknown-4](https://user-images.githubusercontent.com/44592326/208522230-c4066c1f-b0cf-47e3-9b9a-ef92592ce51d.png)

Model Parallelism & Single GPU & Pipeline Parallelism on Small Dataset 

![Unknown](https://user-images.githubusercontent.com/44592326/208522308-fc41c3cf-4d39-4ca7-95b2-1f75e24fd5a5.png)

Model Parallelism & Single GPU & Pipeline Parallelism on CIFAR-10 (Batch Size:32) 

![Unknown-6](https://user-images.githubusercontent.com/44592326/208522408-81d343b4-cc8d-4712-9193-c9d9bc11bfce.png)

Model Parallelism & Single GPU & Pipeline Parallelism on CIFAR-10 (Batch Size:120) 

![Unknown-3](https://user-images.githubusercontent.com/44592326/208522460-5b63872a-21bf-4386-a09d-243f065a05ac.png)

Training Time & Split Size on Pipeline Parallelism

![Unknown-2](https://user-images.githubusercontent.com/44592326/208522561-25dd3f2c-4412-4e6b-be1a-fd498f372175.png)
