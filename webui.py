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
from pynvml import *
from rwkv.utils import PIPELINE, PIPELINE_ARGS    
#nvmlInit()
#gpu_h = nvmlDeviceGetHandleByIndex(0)

#判断设备#
if torch.cuda.is_available():
    device = torch.device("cuda") #调用hip设备(其实写cuda就是hip)
    print("device :hip or cuda")
else: 
    device = torch.device("cpu")

##调用V5模型
title="rwkv_v5_AMD_ROCm"    
model_path = "/home/alic-li/RWKV-LM/model/RWKV-6-v2-ctx4096.roleplay.pth" ##模型路径(可修改)
model = RWKV(model=model_path, strategy='cuda fp16')  ##调整策略
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")  ##模型词库
ctx_limit = 3500
########################## text rwkv ################################################################
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
            
        anser = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in anser:
            out_str += anser
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
 
 
 
################################################dialogue######################################################
model_state = None
Assistant = "Molice"
out_last = 0
occurrence = {}
out_tokens = []
answer = ""
msg = ""
out_str = ""

def chat(
    user_name,
    ctx,
    token_count,
    temperature,
    top_p,
    presencePenalty,
    countPenalty,
):
    global model_state, Assistant, occurrence, answer, msg, out_last, out_str, out_tokens
    if user_name == None:
        user_name = "Bob"

    if msg != "" :
        pass
    #elif os.path.exists("./model-data/" + user_name + ".txt"):      ##启用历史聊天记录分析
        #msg = open("./model-data/" + user_name + ".txt").read()
    elif os.path.exists("./model-data/" + user_name + ".txt"):
        msg = ""
    else:
        msg = open("./model-data/" + user_name + ".txt")   ##新建聊天记录
        msg =""  ##清空当前聊天记录
    
    msg += user_name + ": " + ctx + "\n\n" + Assistant + ": "
    
    if os.path.exists("./model-data/" + user_name + ".pth"):   #加载历史状态
        model_state = torch.load("./model-data/" + user_name + ".pth", map_location=device)
    else:
        model_state = None              ##新建状态
    
    tokens = pipeline.encode(msg)
    out, model_state = model.forward(tokens, model_state)
    
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    
    for i in range(int(token_count)):
        for n in occurrence:
            out[n] -= args.alpha_presence + occurrence[n] * args.alpha_frequency # repetition penalty
        out[0] -= 1e10  # disable END_OF_TEXT
        
        token = pipeline.sample_logits(out, temperature, top_p)
        
        for xxx in occurrence:
            occurrence[xxx] *= 0.99
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        out, model_state = model.forward([token], model_state)
        out_tokens += [token]
        answer += pipeline.decode([token])
        if "\n\n" in answer:
            yield answer.strip()
            break
    if "\n\n" in answer:
            yield answer.strip()
    msg += answer
    text =  user_name + ": " + ctx + "\n\n" + Assistant + ": " + answer
    text_file = open("./model-data/" + user_name + ".txt", "a")
    text_file.write(text)
    torch.save(model_state,"./model-data/" + user_name + ".pth")
    answer = ""
    model_state = None
    gc.collect()    
    torch.cuda.empty_cache()   
    return user_name, model_state, msg
    
################################################save_def#############################################
def save_state():
    global model_state
    torch.save(model_state,"./model-data/" + user_name + ".pth")




################################################Gr_Tab###################################################################
with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>{title}</h1>\n</div>")
    with gr.Tab("续写"):           ##text model tab
        gr.Markdown(f"主程序基于huggingface上的demo,并加之以魔改,作者源代码仓库:(https://github.com/BlinkDL/RWKV-LM) Powered By AMD Radeon!")
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
                user_name = gr.Textbox(lines=1,label="Pleas press you user name~", value="")
                token_count = gr.Slider(10, 10000, label="Max Tokens", step=10, value=333)
                temperature = gr.Slider(0.2, 3.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 10.0, label="Presence Penalty", step=0.1, value=1)
                count_penalty = gr.Slider(0.0, 10.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")                    
                output = gr.Textbox(label="Output", lines=5)               
        submit.click(chat, [user_name,input, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
demo.queue(default_concurrency_limit=6)   #多线程设置
demo.launch(server_name="192.168.0.105", server_port=11451, show_error=True, share=True)

