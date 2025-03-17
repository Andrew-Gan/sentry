export TMPDIR=.
pip install .
pip install torch cuda-python nvidia-dali-cuda120 pillow cupy-cuda12x huggingface_hub

openssl ecparam -name prime256v1 -genkey -noout -out private.pem
openssl ec -in private.pem -pubout -out public.pem

# python3 -m src.sign --model_path /home/torch private-key --private_key private.pem
python3 dali_pipeline.py --model_path /home/torch private-key --private_key private.pem     