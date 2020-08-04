from pathlib import Path
import tarfile

import boto3
from onnxruntime import InferenceSession
from tqdm import tqdm
from transformers import T5Tokenizer

_package_path = Path(__file__).resolve().parent
_models_path = _package_path.joinpath("models")


def get_encoder_decoder_tokenizer():
    """ Function to get a default pre-trained version of T5 in ONNX ready for use """
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


def _progress(t):
    def inner(bytes_amount):
        t.update(bytes_amount)

    return inner


def _download_generation_model(path):
    bucket = 't5-onnx-models'
    key = str(path.name)

    s3 = boto3.client('s3')

    # Gets size of object in bytes
    response = s3.head_object(Bucket=bucket, Key=key)
    size = response['ContentLength']

    # Downloads from S3, reporting progress in bytes to a tqdm progress bar. Units are in bytes. Setting disable to None
    # causes tqdm to check whether the process is attached to a terminal and disable progress bar output if not.
    with tqdm(total=size, unit='B', unit_scale=True, desc=key, disable=None) as t:
        s3.download_file(Bucket=bucket, Key=key, Filename=str(path), Callback=_progress(t))

    # Extracts to model directory
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(path=path.parent)
