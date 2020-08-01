import unittest
import torch

from onnxt5 import create_t5_encoder_decoder, GenerativeT5
from onnxt5.api import get_encoder_decoder_tokenizer


class TestTasks(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        pretrained_model = 't5-base'

        # Loading tokenizer, encoder, and decoder for onnx
        decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()

        # Loading tokenizer, encoder, and decoder for pytorch
        encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_model)
        self.generative_t5_pytorch = GenerativeT5(encoder, decoder_with_lm_head, tokenizer)

        self.generative_t5_onnx = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)

    def test_translation(self):
        prompt = 'translate English to French: I was a victim of a series of accidents.'
        output_pytorch = self.generative_t5_pytorch(prompt, 16, temperature=0.)[0]
        output_onnx = self.generative_t5_onnx(prompt, 16, temperature=0.)[0]
        assert output_pytorch == output_onnx == "Je suis victime d'une série d'accidents."

    def test_summarization(self):
        prompt = '''summarize: Transfer learning, where a model is first pre-trained on a data-rich task before being 
            fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). 
            The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. 
            In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified 
            framework that converts all text-based language problems into a text-to-text format.'''
        output_pytorch = self.generative_t5_pytorch(prompt, 11, temperature=0.)[0]
        output_onnx = self.generative_t5_onnx(prompt, 11, temperature=0.)[0]
        assert output_pytorch == output_onnx == 'a paper explores the landscape of natural language processing'

    def test_q_and_a(self):
        prompt = '''question: What does increased oxygen concentrations in the patient’s
            lungs displace? context: Hyperbaric (high-pressure) medicine uses special
            oxygen chambers to increase the partial pressure of O 2 around the patient
            and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene,
            and decompression sickness (the ’bends’) are sometimes treated using these
            devices. Increased O 2 concentration in the lungs helps to displace carbon
            monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the
            anaerobic bacteria that cause gas gangrene, so increasing its partial pressure
            helps kill them. Decompression sickness occurs in divers who decompress too
            quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and
            helium, forming in their blood. Increasing the pressure of O 2 as soon as
            possible is part of the treatment.'''
        output_pytorch = self.generative_t5_pytorch(prompt, 4, temperature=0.)[0]
        output_onnx = self.generative_t5_onnx(prompt, 4, temperature=0.)[0]
        assert output_pytorch == output_onnx == 'carbon monoxide'