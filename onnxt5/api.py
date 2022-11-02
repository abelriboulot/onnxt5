from pathlib import Path
import tarfile
import os
import glob

import requests
from onnxruntime import InferenceSession
from tqdm import tqdm
from transformers import T5Tokenizer

_package_path = Path(__file__).resolve().parent
_models_path = _package_path.joinpath('model_data')

def get_sess(ouput_prefix):
    """ Function to load previously exported models """
    decoder_sess = InferenceSession(ouput_prefix+"-decoder-with-lm-head.onnx")
    encoder_sess = InferenceSession(ouput_prefix+"-encoder.onnx")
    return decoder_sess, encoder_sess

def get_encoder_decoder_tokenizer():
    """ Function to get a default pre-trained version of T5 in ONNX ready for use """
    try:
        decoder_sess, encoder_sess = _handle_creation_of_sessions()
    except:
        filelist = glob.glob(os.path.join(_models_path, "*"))
        for f in filelist:
            os.remove(f)
        decoder_sess, encoder_sess = _handle_creation_of_sessions()

    # The tokenizer should be the one you trained in the case of fine-tuning
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    return decoder_sess, encoder_sess, tokenizer


def _handle_creation_of_sessions():
    path_t5_encoder = _models_path.joinpath('t5-encoder.onnx')
    path_t5_decoder = _models_path.joinpath('t5-decoder-with-lm-head.onnx')

    _models_path.mkdir(exist_ok=True)

    # Checks if encoder is already expanded
    if not path_t5_encoder.exists():
        path_t5_encoder_tarball = _models_path.joinpath('t5-encoder.tar.gz')
        _download_generation_model(path_t5_encoder_tarball)

    # Checks if decoder is already expanded
    if not path_t5_decoder.exists():
        path_t5_decoder_tarball = _models_path.joinpath('t5-decoder-with-lm-head.tar.gz')
        _download_generation_model(path_t5_decoder_tarball)

    # Loading the models
    decoder_sess = InferenceSession(str(path_t5_decoder))
    encoder_sess = InferenceSession(str(path_t5_encoder))
    return decoder_sess, encoder_sess


def run_embeddings_text(encoder, decoder, tokenizer, prompt, max_context_length=512):
    """ Utility function to get the embeddings of a given text prompt
    Args:
        encoder: inference session to use for the encoder
        decoder: inference session to use for the decoder
        tokenizer: huggingface tokenizer to tokenize the inputs
        prompt: str to run
        max_context_length: maximum number of tokens to use as context

    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').numpy()[:max_context_length]

    # To generate the encoder's last hidden state
    encoder_output = encoder.run(None, {"input_ids": input_ids})[0]
    # To generate the full model's embeddings
    decoder_output = decoder.run(None, {
        "input_ids": input_ids,
        "encoder_hidden_states": encoder_output
    })[0]

    return encoder_output, decoder_output


def _progress(t):
    def inner(bytes_amount):
        t.update(bytes_amount)

    return inner


def _get_size(url):
    response = requests.head(url, allow_redirects=True)
    return int(response.headers['Content-Length'])


def _download_generation_model(path):
    url = f'https://t5-onnx-models.s3.amazonaws.com/{path.name}'
    size = _get_size(url)

    req = requests.get(url, allow_redirects=True, stream=True)

    # Downloads from S3, reporting progress in bytes to a tqdm progress bar. Units are in bytes. Setting disable to None
    # causes tqdm to check whether the process is attached to a terminal and disable progress bar output if not.
    with tqdm(total=size, unit='B', unit_scale=True, desc=str(path.name), disable=None) as t:
        with path.open('wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    t.update(1024)

    # Extracts to model directory
    with tarfile.open(path, "r:gz") as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, path=path.parent)
