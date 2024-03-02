import os


os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
# make sure cuda dir is in the same level as modeling_rwkv.py
from modeling_rwkv import RWKV

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
from rwkv.utils import PIPELINE, PIPELINE_ARGS 
#nvmlInit()
#gpu_h = nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #调用hip设备

ctx_limit = 3500
########################## visual rwkv ################################################################
visual_title = 'ViusualRWKV-v5'
#rwkv_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_rwkv.pth"
#vision_remote_path = "rwkv1b5-vitl336p14-577token_mix665k_visual.pth"
vision_tower_name = 'openai/clip-vit-large-patch14-336'

model_path = "/home/alic-li/RWKV-LM/model/visual.pth"
visual_rwkv = RWKV(model=model_path, strategy='cuda fp16') #调用hip策略
pipeline = PIPELINE(visual_rwkv, "rwkv_vocab_v20230424")  ##模型词库

from modeling_vision import VisionEncoder, VisionEncoderConfig
config = VisionEncoderConfig(n_embd=visual_rwkv.args.n_embd, 
                             vision_tower_name=vision_tower_name, 
                             grid_size=-1)
visual_encoder = VisionEncoder(config)
vision_local_path = "/home/alic-li/RWKV-LM/RWKV-v5/model/visual.pth"
vision_state_dict = torch.load(vision_local_path, map_location='cpu')
visual_encoder.load_state_dict(vision_state_dict)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
visual_encoder = visual_encoder.to(device)
##########################################################################
def visual_generate_prompt(instruction):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    return f"\n{instruction}\n\nAssistant:"

def generate(
    ctx,
    image_state,
    token_count=200,
    temperature=1.0,
    top_p=0.1,
    presencePenalty = 0.0,
    countPenalty = 1.0,
):
    args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.1,
                    alpha_frequency = 1.0,
                    alpha_presence = 0.0,
                    token_ban = [], # ban the generation of some tokens
                    token_stop = [0, 261]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        if i == 0:
            input_ids = pipeline.encode(ctx)[-ctx_limit:]
            out, state = visual_rwkv.forward(tokens=input_ids, state=image_state)
        else:
            input_ids = [token]
            out, state = visual_rwkv.forward(tokens=input_ids, state=state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996        
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    #gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #print(f'{timestamp} - vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()


##########################################################################
cur_dir = os.path.dirname(os.path.abspath(__file__))
visual_examples = [
    [
        f"{cur_dir}/examples_pizza.jpg",
        "What are steps to cook it?"
    ],
    [
        f"{cur_dir}/examples_bluejay.jpg",
        "what is the name of this bird?",
    ],
    [
        f"{cur_dir}/examples_woman_and_dog.png",
        "describe this image",
    ],
]


def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format as needed (JPEG, PNG, etc.)
    # Encodes the image data into base64 format as a bytes object
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_image

image_cache = {}
ln0_weight = visual_rwkv.w['blocks.0.ln0.weight'].to(torch.float32).to(device)
ln0_bias = visual_rwkv.w['blocks.0.ln0.bias'].to(torch.float32).to(device)
def compute_image_state(image):
    base64_image = pil_image_to_base64(image)
    if base64_image in image_cache:
        image_state = image_cache[base64_image]
    else:
        image = image_processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'].to(device)
        image_features = visual_encoder.encode_images(image.unsqueeze(0)).squeeze(0) # [L, D]
        # apply layer norm to image feature, very important
        image_features = F.layer_norm(image_features, 
                                    (image_features.shape[-1],), 
                                    weight=ln0_weight, 
                                    bias=ln0_bias)
        _, image_state = visual_rwkv.forward(embs=image_features, state=None)
        image_cache[base64_image] = image_state
    return image_state

def chatbot(image, question):
    if image is None:
        yield "Please upload an image."
        return
    image_state = compute_image_state(image)
    input_text = visual_generate_prompt(question)
    for output in generate(input_text, image_state):
        yield output
###############GrTab#################
with gr.Blocks(title=visual_title) as demo:
    with gr.Tab("Visual RWKV"):              ##visual model
        gr.Markdown(f"This is the Visual RWKV ")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', label="Image")
            with gr.Column():
                prompt = gr.Textbox(lines=8, label="Prompt", 
                    value="Render a clear and concise summary of the photo.")
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary") 
            with gr.Column():
                output = gr.Textbox(label="Output", lines=10)
            #with gr.Column():
            #    prompt = gr.Textbox(lines=2, label="Prompt", value="Assistant: Sure! Here is a very detailed plan to create flying pigs:")
            #    token_count = gr.Slider(10, 333, label="Max Tokens", step=10, value=333)
            #    temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
            #    top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
            #    presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0)
            #    count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
        data = gr.Dataset(components=[image, prompt], samples=visual_examples, label="Examples", headers=["Image", "Prompt"])
        submit.click(chatbot, [image, prompt], [output])
        ##submit.click(chatbot, [image, prompt,token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [image, prompt])

demo.queue(concurrency_count=1, max_size=10)
demo.launch(sever_name="127.0.0.0",sever_port="8080",show_error=True,share=True)