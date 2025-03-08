FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip
RUN pip install --break-system-packages typing_extensions protobuf \
    sigstore sigstore_protobuf_specs cuda-python in_toto_attestation \
    cryptography certifi pyopenssl huggingface_hub transformers sentencepiece \
    sacremoses torch nvidia-dali-cuda120 pillow

ENV TORCH_HOME=/home/torch
ENV DALI_EXTRA_PATH="/home/DALI_extra"

COPY main.py /home/
COPY src /home/src

WORKDIR /home
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# WORKDIR /home/src
# CMD ["python3", "sign.py", "--model_path", "/home/torch", "private-key", "--private_key", "../private.pem"]

WORKDIR /home
CMD ["python3", "main.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
