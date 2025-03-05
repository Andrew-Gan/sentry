FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt update && apt install -y python3-pip
RUN pip install torch nvidia-dali-cuda120 pillow

COPY test_dali.py /home/
WORKDIR /home

ENV TORCH_HOME=/home/torch
ENV DALI_EXTRA_PATH="/home/DALI_extra"

CMD ["python3", "test_dali.py"]
