ARG CUDA_VERSION="10.0"
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu18.04
MAINTAINER pingjunchen <pingjunchen@ieee.org>
ARG USER_NAME="pchen6"

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential curl sudo wget vim \
  && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it

RUN adduser --disabled-password --gecos '' --shell /bin/bash ${USER_NAME}
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN chmod 777 /home/${USER_NAME}
WORKDIR /home/${USER_NAME}

# Install Miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && bash ~/Miniconda3-py38_4.8.2-Linux-x86_64.sh -p ~/miniconda -b \
 && rm ~/Miniconda3-py38_4.8.2-Linux-x86_64.sh
ENV PATH=/home/${USER_NAME}/miniconda/bin:$PATH
## Create a Python 3.8.3 environment
RUN /home/${USER_NAME}/miniconda/bin/conda install conda-build \
 && /home/${USER_NAME}/miniconda/bin/conda create -y --name py38 python=3.8.3 \
 && /home/${USER_NAME}/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/home/${USER_NAME}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Python packages installation
## Using pip
RUN pip install numpy==1.19.2
RUN pip install gpustat==0.6.0
## Install pytorch
RUN conda install pytorch torchvision cudatoolkit=${CUDA_VERSION} -c pytorch

WORKDIR /home/${USER_NAME}/HelloDL
ADD *.py /home/${USER_NAME}/HelloDL/
