FROM tensorflow/tensorflow:latest-gpu 

# Update the package index and install some necessary packages

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        build-essential \
        cmake \
        git \
        curl\
        wget


WORKDIR /tmp

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update 
RUN apt-get update && \
  apt-get install -y \
        cuda-drivers-550\
        nvidia-container-toolkit
RUN nvidia-ctk runtime configure --runtime=docker  




# Install Python 3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.10 python3-pip
# RUN pip3 install --upgrade pip   

# set WORKDIR
WORKDIR /home/root/rl_ws

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# replace mirror

# RUN apt-get update
# RUN apt-get install -q -y --no-install-recommends wget \
# && wget http://archive.ubuntu.com/ubuntu/pool/main/c/ca-certificates/ca-certificates_20240203ubuntu0.22.04.1_all.deb \
# && dpkg -i ./ca-certificates_20240203ubuntu0.22.04.1_all.deb

# RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
#   echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse\n" > /etc/apt/sources.list && \
#   echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse\n" >> /etc/apt/sources.list && \
#   echo "deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse\n" >> /etc/apt/sources.list && \
#   echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse\n" >> /etc/apt/sources.list

# http://archive.ubuntu.com/ubuntu/pool/main/c/ca-certificates/ca-certificates_20240203~22.04.1_all.deb



# dependencies for gym
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
#  gcc-8 \
#  g++-8 \
#  vulkan-utils \
#  mesa-vulkan-drivers \
#  pigz \
 git \
 libegl1 \
 git-lfs \
 libsm6 \
 xauth \
 x11-apps

# pip install requirements
COPY requirement.txt .
RUN pip install -r requirement.txt


RUN ln -sfn /usr/bin/python3.10 /usr/bin/python


RUN apt-get update && \
    apt-get install -y \
        g++\
        libglib2.0-dev\
        cmake\
        libboost-all-dev\
        python3-dev
       
RUN wget https://github.com/lcm-proj/lcm/releases/download/v1.4.0/lcm-1.4.0.zip && unzip lcm-1.4.0.zip && mv lcm-1.4.0 lcm
WORKDIR /home/root/rl_ws/lcm
RUN mkdir -p /home/root/rl_ws/lcm/build
WORKDIR /home/root/rl_ws/lcm/build
RUN cmake .. 
RUN make 
RUN make install

WORKDIR /home/root/rl_ws/lcm/lcm-python
RUN  python3 setup.py install

WORKDIR /home/root/rl_ws
RUN apt-get update && \
    apt-get install -y \
        libmsgpack*



# RUN chmod +x setup.sh

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
