FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip

WORKDIR /home
COPY requirements.txt /home/requirements.txt
RUN pip install -r requirements.txt

ENV TORCH_HOME=/home/torch

COPY main.py /home/main.py
COPY src /home/src
COPY RapidEC /home/RapidEC
COPY dataset_formatter /home/dataset_formatter

RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# CMD ["python3", "-m", "src.sign", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
CMD ["python3", "main.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
# CMD ["tail", "-f"]
