FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  curl \
  tmux \
  wget \
  git \
  cmake \
  vim \
  pkg-config \
  unzip \
  libgtk2.0-dev \
  imagemagick \
  graphviz && \
  mkdir /projects

COPY . /projects/llm_agents_chemistry/multi_agents_system

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    export PATH=~/miniconda3/miniconda/bin:$PATH && \
    ~/miniconda3/bin/conda init bash && \
    ~/miniconda3/bin/conda init zsh


ENV PATH ~/miniconda3/miniconda/bin:$PATH
ENV LB_LIBRARY_PATH="~/miniconda3/miniconda/lib:\$LB_LIBRARY_PATH"
SHELL ["/bin/bash", "-ic"]
RUN conda update -n base conda && \
    conda update --all -y

RUN cd /projects/llm_agents_chemistry/multi_agents_system && \
    conda create -n llm_pipeline python=3.11.10 && \
    conda deactivate && \
    conda activate llm_pipeline && \ 
    conda install conda-libmamba-solver && \
    pip install --no-cache-dir -r /projects/llm_agents_chemistry/multi_agents_system/requirements.txt && \
    curl -fsSL https://ollama.com/install.sh > install.sh &&\
    bash install.sh

ENV PYTHONPATH="${PYTHONPATH}:/projects/llm_agents_chemistry"
WORKDIR /projects/llm_agents_chemistry

ENTRYPOINT [ "/bin/bash", "multi_agents_system/run.sh" ]

