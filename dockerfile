FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip gcc-multilib
RUN pip install --break-system-packages typing_extensions protobuf \
    sigstore sigstore_protobuf_specs cuda-python in_toto_attestation \
    cryptography certifi pyopenssl huggingface_hub transformers sentencepiece \
    sacremoses

ENV TORCH_HOME=/home/torch

COPY src /home/src
WORKDIR /home/src
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# CMD ["tail", "-f", "/dev/null"]
CMD ["python3", "sign.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]

# WORKDIR /home/src/model_signing/cuda
# CMD ["nvcc", "ltHash.cu", "-o", "hash"]
