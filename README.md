# ONNX-T5
T5 Implementation in ONNX with utility functions for fast inference. This package is still in alpha
stage, therefore some functionalities such as beam searches are still in development.

## Installation

## Usage
The simplest way to get started for generation is to use the default pre-trained
version of T5 on ONNX included in the package
```python
from onnxt5 import GenerativeT5
from onnxt5.api import get_encoder_decoder_tokenizer
decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
prompt = 'translate English to French: I was a victim of a series of accidents.'

output_text, output_logits = generative_t5('translate English to French: I was a victim of a series of accidents.', 21, temperature=0.)
# output_text: "Je suis victime d'une série d'accidents."
```

If you want to get the embeddings of text, you can run the following
```python
from onnxt5.api import get_encoder_decoder_tokenizer, run_embeddings_text

decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
prompt = 'Listen, Billy Pilgrim has come unstuck in time.'
encoder_embeddings, decoder_embeddings = run_embeddings_text(encoder_sess, decoder_sess, tokenizer, prompt)
```

ONNXT5 also lets you export and use your own models. See the `examples\` folder for more detailed examples

## Functionalities
* Export your own T5 models to ONNX
* 2x speed of inference


## Benchmarks