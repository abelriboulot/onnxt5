from onnxruntime import InferenceSession
from transformers import T5Tokenizer
from onnxt5.api import get_encoder_decoder_tokenizer, run_embeddings_text, get_sess

# The easiest way is to use the onnxt5 api and load the default pre-trained version of t5
decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()

# You can load pre-exported models with get_sess (do note you need the tokenizer you trained also)
# decoder_sess, encoder_sess = get_sess(output_path)

# You can also load model_data manually:
# decoder_sess = InferenceSession('/home/abel/t5-decoder-with-lm-head.onnx')
# encoder_sess = InferenceSession('/home/abel/t5-encoder.onnx')
# The tokenizer should be the one you trained in the case of fine-tuning
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

prompt = 'Listen, Billy Pilgrim has come unstuck in time.'
# To get embeddings you can either use our utility function
encoder_embeddings, decoder_embeddings = run_embeddings_text(encoder_sess, decoder_sess, tokenizer, prompt)

# Or do it manually as follow
input_ids = tokenizer.encode(prompt, return_tensors='pt').numpy()

# To generate the encoder's last hidden state
encoder_output = encoder_sess.run(None, {"input_ids": input_ids})[0]
# To generate the full model's embeddings
decoder_output = decoder_sess.run(None, {
                                        "input_ids": input_ids,
                                        "encoder_hidden_states": encoder_output
    })[0]