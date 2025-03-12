FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt update && apt install -y nvidia-gds
CMD ["/usr/local/cuda-12.8/gds/tools/gdscheck.py", "-p"]
