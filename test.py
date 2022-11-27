import os
import json
import math
import numpy as np
import torch
import argparse
import soundfile as sf

import commons
import utils
from models_comp import SynthesizerTrnComp
from text.symbols import symbols
from text import text_to_generalized_sequence
from text import _symbol_to_id,_id_to_symbol

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', default='SIL^bba^bba^SIL', type=str, help='This is name.')
    parser.add_argument('-o', '--output', default='hoge.wav', type=str, help='This is name.')

    return parser.parse_args()
    
def get_text(text, hps):
    text_norm = text_to_generalized_sequence(text, ['symbol_cleaners'])
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return text_norm

def generate(net_g,id_list,w_ceil=None):
  with torch.no_grad():
      x_tst_lengths = torch.LongTensor([len(id_list)]).cuda()
      if w_ceil is None:
        audio, _, _, _ = net_g.inferComp(id_list, x_tst_lengths, noise_scale=0.2, noise_scale_w=0.5, length_scale=1)
      else:
        audio, _, _, _ = net_g.inferComp(id_list, x_tst_lengths, noise_scale=0.2, noise_scale_w=0.5, length_scale=1, w_ceil=torch.ceil(w_ceil))
      audio = audio[0,0].data.cpu().float().numpy()
  return audio

def main():
    args = get_args()
    hps = utils.get_hparams_from_file("./logs/bird01_cath/config.json")
    net_g = SynthesizerTrnComp(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint("./logs/bird01_cath/G_9000.pth", net_g, None)

    audio = generate(net_g, get_text(args.text, hps))
    sf.write(args.output, audio, hps.data.sampling_rate, format="wav")

if __name__ == '__main__':
    main()