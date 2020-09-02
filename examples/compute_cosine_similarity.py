# This small example shows how to compute cosine similarity of the averaged of two sentences
import torch
from torch.nn import CosineSimilarity
from onnxt5.api import get_encoder_decoder_tokenizer, run_embeddings_text

decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
prompt_1 = 'The ocean is deep'
prompt_2 = 'The sea is large'

encoder_embeddings_1, decoder_embeddings_1 = run_embeddings_text(encoder_sess, decoder_sess, tokenizer, prompt_1)
encoder_embeddings_2, decoder_embeddings_2 = run_embeddings_text(encoder_sess, decoder_sess, tokenizer, prompt_2)


cosine_similarity = CosineSimilarity(1)(torch.tensor(encoder_embeddings_1.mean(1)),
                    torch.tensor(encoder_embeddings_2.mean(1)))

print(f"Cosine similarity: {cosine_similarity}")