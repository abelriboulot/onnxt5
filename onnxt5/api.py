import os
import tarfile
from onnxruntime import InferenceSession
from transformers import T5Tokenizer
import requests

import onnxt5


def get_encoder_decoder_tokenizer():
    """ Function to get a default pre-trained version of T5 in ONNX ready for use """
    package_path = os.path.dirname(onnxt5.__file__)
    path_t5_encoder = os.path.join(package_path, 'model_data', 't5-encoder.onnx')
    path_t5_decoder = os.path.join(package_path, 'model_data', 't5-decoder-with-lm-head.onnx')

    # Checks if encoder is already expanded
    if not os.path.exists(path_t5_encoder):
        download_generation_model(os.path.join(package_path, 'model_data'), 't5-encoder.tar.gz')

    # Checks if decoder is already expanded
    if not os.path.exists(path_t5_decoder):
        download_generation_model(os.path.join(package_path, 'model_data'), 't5-decoder-with-lm-head.tar.gz')

    # Loading the model_data
    decoder_sess = InferenceSession(path_t5_decoder)
    encoder_sess = InferenceSession(path_t5_encoder)
    # The tokenizer should be the one you trained in the case of fine-tuning
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    return decoder_sess, encoder_sess, tokenizer

def run_embeddings_text(encoder, decoder, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').numpy()

    # To generate the encoder's last hidden state
    encoder_output = encoder.run(None, {"input_ids": input_ids})[0]
    # To generate the full model's embeddings
    decoder_output = decoder.run(None, {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_output
    })[0]

    return encoder_output, decoder_output

def download_generation_model(output_dir, object):
    url = f'https://t5-onnx-models.s3.amazonaws.com/{object}'
    r = requests.get(url, allow_redirects=True)
    downloaded_file_path = os.path.join(output_dir, object)
    open(downloaded_file_path, 'wb').write(r.content)
    tar = tarfile.open(downloaded_file_path, "r:gz")
    tar.extractall(path=output_dir)
    tar.close()
