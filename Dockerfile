FROM nvidia/cuda:10.2-cudnn8-runtime
ARG PYTHON=python3.6.2

WORKDIR /workspace

# Install basic development tools
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   vim \
                   ${PYTHON} \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install torch 
RUN python3 -m pip install click 
RUN python3 -m pip install numpy 
RUN python3 -m pip install einops 
RUN python3 -m pip install wandb 
RUN python3 -m pip install scikit-learn 
RUN python3 -m pip install python-hostlist 
RUN python3 -m pip install tqdm 
RUN python3 -m pip install requests 
RUN python3 -m pip install pyyaml 
RUN python3 -m pip install timm==0.4.12 
RUN python3 -m pip install mmcv==1.3.8 
RUN python3 -m pip install mmsegmentation==0.14.1

COPY . .

CMD ["/bin/bash"]