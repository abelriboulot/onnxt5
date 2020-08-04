from transformers import T5Tokenizer
from onnxt5 import generate_onnx_representation, GenerativeT5
from onnxruntime import InferenceSession

# Export default pretrained version
generate_onnx_representation(pretrained_version='t5-small', output_prefix='/home/abel/t5')

# To export a fine-tuned version, the same method works, but give the path to your version
generate_onnx_representation(pretrained_version='/home/abel/t5-small.bin', output_prefix='/home/kurt/t5')

# This will generate two files:
# - /home/kurt/t5-decoder-with-lm-head.onnx >>> the decoder
# - /home/kurt/t5-encoder.onnx >>> the encoder

# Loading the model_data
decoder_sess = InferenceSession('/home/abel/t5-decoder-with-lm-head.onnx')
encoder_sess = InferenceSession('/home/abel/t5-encoder.onnx')
# The tokenizer should be the one you trained in the case of fine-tuning
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Generating text
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
generative_t5('translate English to French: I was a victim of a series of accidents.', 21, temperature=0.)[0]
# Output: "Je suis victime d'une série d'accidents."
