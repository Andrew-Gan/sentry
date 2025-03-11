FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip nvidia-cuda-toolkit
RUN pip install --break-system-packages typing_extensions protobuf \
    sigstore sigstore_protobuf_specs cuda-python in_toto_attestation \
    cryptography certifi pyopenssl huggingface_hub transformers sentencepiece \
    sacremoses torch nvidia-dali-cuda120 pillow cupy

ENV TORCH_HOME=/home/torch
ENV DALI_EXTRA_PATH="/home/DALI_extra"
ENV DALI_GDS_CHUNK_SIZE=4096

COPY dali_pipeline.py /home/
COPY src /home/src

WORKDIR /home
RUN openssl ecparam -name prime256v1 -genkey -noout -out private.pem
RUN openssl ec -in private.pem -pubout -out public.pem

# CMD ["python3", "-m", "src.sign", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
CMD ["python3", "dali_pipeline.py", "--model_path", "/home/torch", "private-key", "--private_key", "private.pem"]
# CMD ["tail", "-f"]
