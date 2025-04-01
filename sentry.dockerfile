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
COPY main.py /home/main.py

WORKDIR /home
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# CMD ["python3", "-m", "src.sign", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
CMD ["python3", "main.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
# CMD ["tail", "-f"]
