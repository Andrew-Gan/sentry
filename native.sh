pip install -r requirements.txt

openssl ecparam -name prime256v1 -genkey -noout -out private.pem
openssl ec -in private.pem -pubout -out public.pem

export CUFILE_ENV_PATH_JSON=cufile.json

# python3 -m src.sign --model_path /home/torch private-key --private_key private.pem
python3 main.py --model_path /home/torch private-key --private_key private.pem
