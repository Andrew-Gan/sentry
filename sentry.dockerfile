FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y g++-multilib python3-pip libnuma1 udev

COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt

ENV TORCH_HOME=/home/torch
ENV CUFILE_ENV_PATH_JSON=/home/cufile.json

COPY RapidEC /home/RapidEC
WORKDIR /home/RapidEC
RUN nvcc -Xcompiler '-fPIC' -o gsv.so -shared gsv.cu

COPY dataset_formatter /home/dataset_formatter
COPY cufile.json /home/cufile.json
COPY src /home/src
COPY *.py /home

WORKDIR /home
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem
