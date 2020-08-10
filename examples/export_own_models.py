# This is a very small notebook showing how to grab a pre-trained T5 model, fine-tune it, and export it to onnx.]
# A lot of this is inspired by huggingface.

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, AdamW
import torch
from onnxt5 import generate_onnx_representation, GenerativeT5
from onnxt5.api import get_sess
import requests

base_model = "t5-base"

# Setting up the model and tokenizer
config = T5Config.from_pretrained(base_model)
config.n_positions = 256 # You can change the properties of your model here
model = T5ForConditionalGeneration(config=config)

# Download vocab file
tokenizer = T5Tokenizer(config=config, vocab_file="test_sentencepiece.model")
model.train()

# Let's setup our optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# An example of a positive and negative sentence (please please please use more than one example :) )
text_batch = ["sentiment: Things are pretty aight", "sentiment: Things are dire."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

for i in range(10):
    # Setting up targets
    labels = torch.tensor([tokenizer.encode("1")[1], tokenizer.encode("0")[1]]).unsqueeze(1)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    # Running the backpropagation
    loss = outputs[0]
    loss.backward()
    optimizer.step()

model.eval()

# Export to ONNX
generate_onnx_representation(output_prefix="/Users/abelriboulot/t5-own-", model=model)

# Load the model
decoder_sess, encoder_sess = get_sess("/Users/abelriboulot/t5-own-")
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
print(generative_t5("sentiment: Things are pretty aight", 1, temperature=0.)[0])
# Output: 1 <<< Positive