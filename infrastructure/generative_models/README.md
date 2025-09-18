# Instructions for build and run container with generative models

The easiest way to work with this part of the project is to build a container on a server with an available video card.

```
git clone https://github.com/ITMO-NSS-team/CoScientist.git

cd infrastructure/generative_models

docker build -t generative_model_backend .

```
# Running a container

The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.

Next, after you have created an image on your server (or locally), you need to run the container with the command:
```
docker run --name molecule_generator --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<your device ID> -it --init generative_model_backend:latest bash

OR 

docker run --name molecule_generator --runtime=nvidia -e --gpus all -it --init generative_model_backend:latest bash
```