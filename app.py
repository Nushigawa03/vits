import streamlit as st
st.title("FakeBird demo")

import matplotlib.pyplot as plt

import numpy as np
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models_comp import SynthesizerTrnComp
from text.symbols import symbols
from text import text_to_generalized_sequence

from scipy.io.wavfile import write

import glob
import tempfile
import os
import base64

import soundfile as sf
import streamlit as st

from streamlit_elements import elements, mui, html, sync

MAX_PHRASE_LENGTH=10
info_obj=None


def get_binary_file_downloader_html(bin_file, file_label='File', extension=""):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}{extension}">Download {file_label}</a>'
    return href

def get_text(text, hps):
    text_norm = text_to_generalized_sequence(text, ['symbol_cleaners'])
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    print(text_norm)
    return text_norm

def search_true(txt,filename="./bird01_cath_valid.txt"):
    wav_list=[]
    for line in open(filename):
        arr=line.strip().split("|")
        if arr[1]==txt:
            wav_list.append(arr[0])
    wav_filename=np.random.choice(wav_list)
    st.audio(wav_filename)
    #print(wav_filename)
    #ipd.display(ipd.Audio("data_bird01_cath_phrase/2012-02-09_10-30-00-000000.wav.19275621-19290577.wav"))



def do(config_path,ckpt_path,txt,info_obj):
    ####
    if txt is None and info_obj is not None:
        mat=np.array(info_obj["mat"])
        tr_syms=info_obj["symbols"]

        state=0
        states=[0]
        for _ in range(30):
            next_state=np.random.choice(len(tr_syms), 1, p=mat[state,:])[0]
            states.append(next_state)
            state=next_state
            if next_state==1:
                break

        txt="^".join([tr_syms[i] for i in states])

    ####
    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrnComp(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)#.cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(ckpt_path, net_g, None)
    

    # txt = 'SIL^ala^{"ai":0.5,"alc":0.5}^ala^{"ai":3,"hi":-1}^SIL'
    x_tst = get_text(txt, hps)
    with torch.no_grad():
        x_tst_lengths = torch.LongTensor([len(x_tst)])#.cuda()
        audio = net_g.inferComp(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    
    st.session_state.results.append({
        "audio":audio,
        "txt":txt,
        "sampling_rate":hps.data.sampling_rate})

start_flag=False
with st.sidebar:
    button_css = """
                <style>
                .labelButton {
                  text-transform: none;
                }
                </style>
                """
    st.markdown(button_css, unsafe_allow_html=True)


    base_log_path='./logs'
    log_list=os.listdir(path=base_log_path)

    mode = st.selectbox(
        'How would you like to be contacted?',
         log_list)
    if "mode" not in st.session_state:
        st.session_state["mode"]=mode
        info_path=mode+"_info.json"
        info_obj=json.load(open(info_path))
        st.session_state["info_obj"]=info_obj

    if st.session_state["mode"]!=mode:
        info_path=mode+"_info.json"
        st.session_state["mode"]=mode
        info_obj=json.load(open(info_path))
        st.session_state["info_obj"]=info_obj
    else:
        info_obj=st.session_state["info_obj"]

    symbol_list=info_obj["symbols"][2:]
    #st.write('You selected:', mode)
    config_path=base_log_path+"/"+mode+"/config.json"
    #st.write('config:', config_path)

    info_path=mode+"_info.json"
    info_obj=json.load(open(info_path))
    symbol_list=info_obj["symbols"][2:]



    ckpt_list=glob.glob(base_log_path+"/"+mode+"/G_*.pth")
    sel_ckpt = st.selectbox(
        'select checkpoint?',
         [os.path.basename(e) for e in ckpt_list])

    with st.expander("Custom phrase:"):
        
        st.write("Phrase:")
        if "phrase_text" not in st.session_state:
            st.session_state.phrase_text = []
        if "phrase_text_D" not in st.session_state:
            st.session_state.phrase_text_D = {}
        phrase_text=st.session_state.phrase_text
        phrase_text_D=st.session_state.phrase_text_D
        phrase_placeholder = st.empty()
        s = st.selectbox(
            'Select an syllable and push Add button:',
             symbol_list)
        
        st.write("────────────────")
        st.write("Custom syllable:")
        generalized_syllable_placeholder = st.empty()
        gs = st.selectbox(
            'Select an syllable and push Mix button:',
             symbol_list)
        gn = st.number_input(
            'Weight',
            value=1.0,
            )
        if st.button('Mix'):
            phrase_text_D[gs]=gn
        # if st.button('Add'):
        st.write("────────────────")
        if st.button('Add'):
            if phrase_text_D != {}:
                phrase_text.append(json.dumps(phrase_text_D))
                phrase_text_D = {}
            else:phrase_text.append(s)
            

        new_phrase_text=[]
        with phrase_placeholder.container():
            with elements("multiple_children"):
                for i in range(MAX_PHRASE_LENGTH):
                    key="evt"+str(i)
                    if key not in st.session_state:
                        st.session_state[key] = None

                    if st.session_state[key] is not None:
                        obj = st.session_state[key]
                        st.session_state[key]=None

                    else:
                        if i<len(phrase_text):
                            s=phrase_text[i]
                            new_phrase_text.append(s)
                            mui.Button(
                                #mui.icon.ArrowForwardIos,
                                mui.Typography(s),
                                mui.icon.ClearOutlined,
                                variant="contained",
                                className="labelButton",
                                onClick=sync(key)
                            )

        new_phrase_text_D={}

        with generalized_syllable_placeholder.container():
            with elements("multiple_syllable_vectors"):
                for k in phrase_text_D.keys():
                    key="evtK"+str(k)
                    if key not in st.session_state:
                        st.session_state[key] = None

                    if st.session_state[key] is not None:
                        obj = st.session_state[key]
                        st.session_state[key]=None
                    else:
                        new_phrase_text_D[k] = phrase_text_D[k]
                        mui.Button(
                            #mui.icon.ArrowForwardIos,
                            mui.Typography(str(k)+':'+str(phrase_text_D[k])),
                            mui.icon.ClearOutlined,
                            variant="contained",
                            className="labelButton",
                            onClick=sync(key)
                        )
        
        st.session_state.phrase_text=new_phrase_text
        st.session_state.phrase_text_D=new_phrase_text_D


    if sel_ckpt is not None:
        ckpt_path=base_log_path+"/"+mode+"/"+sel_ckpt
        if st.button('generate'):
            start_flag=True

if "results" not in st.session_state:
    st.session_state.results = []
if start_flag:
    txt=None
    if len(st.session_state.phrase_text)>0:
        txt="SIL^"+"^".join(st.session_state.phrase_text)+"^SIL"
    do(config_path,ckpt_path,txt,info_obj)


for el in st.session_state.results:
    fp = tempfile.NamedTemporaryFile(delete=False)
    sf.write(fp.name, el["audio"], el["sampling_rate"], format="wav")
    st.write(el["txt"].upper())
    st.audio(fp.name)
    fp.close()
    os.remove(fp.name)
