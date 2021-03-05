FROM nvidia/cuda:10.1-base-ubuntu18.04
MAINTAINER pingjunchen <pingjunchen@ieee.org>


RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential curl sudo wget vim \
  && rm -rf /var/lib/apt/lists/*


# Install Miniconda
WORKDIR /App
RUN chmod 777 /App
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && bash Miniconda3-py38_4.8.2-Linux-x86_64.sh -p /App/miniconda -b
ENV PATH=/App/miniconda/bin:$PATH
## Create a Python 3.8.3 environment
RUN /App/miniconda/bin/conda install conda-build \
 && /App/miniconda/bin/conda create -y --name py38 python=3.8.3 \
 && /App/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/App/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH


# Python packages installation
## Using pip
RUN pip install requests==2.25.1 numpy==1.19.2 gpustat==0.6.0
## Install pytorch
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch


## Create a non-root user and switch to it
#ARG USER_NAME="ping"
#RUN adduser --disabled-password --gecos '' --shell /bin/bash ${USER_NAME}
#RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
#USER ${USER_NAME}
#ENV HOME=/home/${USER_NAME}
#RUN chmod 777 /home/${USER_NAME}
#WORKDIR /home/${USER_NAME}
