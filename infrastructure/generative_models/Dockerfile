FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04


ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG HF_TOK
ARG GITHUB_TOKEN




RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential curl wget git cmake vim pkg-config unzip \
    libgtk2.0-dev imagemagick graphviz && \
    mkdir /projects && \
    rm -rf /var/lib/apt/lists/*


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



# Then proceed with updates
RUN conda config --set plugins.auto_accept_tos yes && \
    conda update -n base conda -y && \
    conda update --all -y


RUN conda create -n Mol_gen_env python=3.8 && \
    conda deactivate && \
    conda activate Mol_gen_env && \ 
    conda install conda-libmamba-solver && \
    conda install -c "conda-forge/label/cf202003" openbabel --solver libmamba
    #pip install -r requirements.txts
    #pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

#Change github repo !!!

RUN cd /projects &&\
    git clone https://${GITHUB_TOKEN}@github.com/alinzh/MADD.git 
#COPY . /projects/generative_models
WORKDIR /projects/MADD/infrastructure/generative_models

RUN ~/miniconda3/envs/Mol_gen_env/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN ~/miniconda3/envs/Mol_gen_env/bin/pip install -r requirements.txt

RUN ~/miniconda3/envs/Mol_gen_env/bin/huggingface-cli download SoloWayG/Molecule_transformer --local-dir autotrain/

# Setting environment variables
ENV ML_MODEL_URL='http://10.64.4.254:81'
ENV HF_TOKEN=$HF_TOK
ENV LD_LIBRARY_PATH="/opt/conda/envs/Mol_gen_env/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/projects/MADD/infrastructure/generative_models:$PYTHONPATH"
ENV PYTHONPATH="/projects/MADD/:$PYTHONPATH"
ENV PYTHONPATH="/projects/MADD/infrastructure:$PYTHONPATH"
ENV GEN_APP_PORT='293'

# Run
RUN chmod +x /projects/MADD/infrastructure/generative_models/api.sh
#ENTRYPOINT ["/bin/bash", "/projects/MADD/infrastructure/generative_models/api.sh"]