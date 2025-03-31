FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip
RUN pip install --break-system-packages typing_extensions protobuf \
    sigstore sigstore_protobuf_specs in_toto_attestation \
    cryptography certifi pyopenssl huggingface_hub transformers sentencepiece \
    sacremoses torch cuda-python nvidia-dali-cuda120 pillow cupy datasets

ENV TORCH_HOME=/home/torch

COPY main.py /home/
COPY src /home/src
COPY RapidEC /home/RapidEC
COPY dataset_formatter /home/dataset_formatter

WORKDIR /home
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# CMD ["python3", "-m", "src.sign", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
CMD ["python3", "main.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
# CMD ["tail", "-f"]
