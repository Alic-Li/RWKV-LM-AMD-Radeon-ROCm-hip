import os


os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from src.modeling_rwkv import RWKV

import gc
import gradio as gr
import base64
from io import BytesIO
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import CLIPImageProcessor
#from huggingface_hub import hf_hub_download
from pynvml import *
#nvmlInit()
#gpu_h = nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda") #调用hip设备(其实写cuda就是hip)
if torch.cuda.is_available():
    print("device :hip or cuda")
CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)
ctx_limit = 3500
########################## text rwkv ################################################################

from rwkv.utils import PIPELINE, PIPELINE_ARGS     
"""
title_v6 = "RWKV-x060-World-1B6-v2-20240208-ctx4096"     ##调用V6模型
model_path_v6 = hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename=f"{title_v6}.pth")
model_v6 = RWKV(model=model_path_v6, strategy= cuda fp16')
pipeline_v6 = PIPELINE(model_v6, "rwkv_vocab_v20230424")  """
##调用V5模型
title="rwkv_v5_AMD_ROCm"    
model_path = "/home/alic-li/RWKV-LM/model/world.pth" ##模型路径(可修改)
model = RWKV(model=model_path, strategy='cuda fp16')  ##调整策略
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  ##模型词库
user = "user"

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"please input something~~~"

model_tokens = []
model_state = None

def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        input_ids = pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token]
        out, state = model.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
            
        ttt = pipeline.decode([token])
        www = 1
        if ttt in ' \t0123456789':
            www = 0
        #elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
        #    www = 0.5
        if token not in occurrence:
            occurrence[token] = www
        else:
            occurrence[token] += www
            
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    #gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')       #显示GPU占用，希望可以找到调用RadeonGPU占用的库
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()
 
 
 
################################################dialogue###################################################################
def chat(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    model_tokens = []
    model_state = None

    ctx = ctx.replace("\r\n", "\n")

    tokens = pipeline.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]
        occurrence = {}
        out_tokens = []
        out_last = 0
        out_str = ''

        for i in range(int(token_count)):
            for n in occurrence:
                out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency # repetition penalty
            out[0] -= 1e10  # disable END_OF_TEXT

            token = pipeline.sample_logits(out, temperature, top_p)

            out, model_state = model.forward([token], model_state)
            model_tokens += [token]

            out_tokens += [token]

            for xxx in occurrence:
                occurrence[xxx] *= countPenalty
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = pipeline.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                out_str += tmp
                yield out_str.strip()
                out_last = i + 1

            if "\n\n" in tmp:
                out_str += tmp
                yield out_str.strip()
                break

gc.collect()
torch.cuda.empty_cache()    
    



################################################Gr_Tab###################################################################
with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title}</h1>\n</div>")
    with gr.Tab("续写"):           ##text model tab
        gr.Markdown(f"主程序基于huggingface上的demo,作者源代码仓库:(https://github.com/BlinkDL/RWKV-LM) Powered By AMD Radeon!")
        gr.Markdown(f"######玉子姐姐最可爱了～～～######")
        gr.Markdown(f"######模型被调教坏了从我显卡上滚出去！要被玩坏的～～～######")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
    with gr.Tab("chat"):           ##text model tab
        gr.Markdown(f"######玉子姐姐最可爱了～～～######")
        gr.Markdown(f"######模型被调教坏了从我显卡上滚出去！要被玩坏的～～～######")
        with gr.Row():
            with gr.Column():
                input = gr.Textbox(lines=2, label="input", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=1)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        submit.click(chat, [input, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])

##demo.queue(concurrency_count=1, max_size=10)   #多线程设置
demo.launch(server_name="127.0.0.1",server_port=8080,show_error=True,share=False)
