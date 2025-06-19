"""
Sample code to load the test prompts and save the generated images.
You can modify the code your as your needs.

TODO:
1. Load your trained model.
2. Load the pretrained module as train.py.
3. Generate 1000 results using the prompts in test.json.
"""

import json, math, os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler 
from diffusers import DPMSolverMultistepScheduler 
from transformers import CLIPTokenizer, CLIPTextModel

UNET_CKPT = "outputs/ckpt/unet_Epoch260/pytorch_model.bin"  
BATCH_SIZE = 32
NUM_STEPS  = 130      
GUIDANCE   = 8.0       # CFG scale
SAVE_DIR   = Path("results")
SAVE_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UNet 架構設定（必須與 train.py 一致）
unet_cfg = dict(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    cross_attention_dim=512,          # 與 CLIP hidden_size 對齊
    layers_per_block=2,
    block_out_channels=(256, 384, 512, 768),
    down_block_types=(
        "CrossAttnDownBlock2D", "DownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D", "UpBlock2D",
        "CrossAttnUpBlock2D", "UpBlock2D",
    ),
)
# ===================================== #

@torch.no_grad()
def sample_batch(prompts: List[str], unet, vae,
                 tokenizer, text_encoder, scheduler,
                 device) -> torch.Tensor:
    """
    由 prompt list 產生一批影像 (tensor 0~1)；N = len(prompts)
    """
    n = len(prompts)

    # --- CLIP 編碼 prompt 文字 ---
    cond_ids = tokenizer(prompts, padding="max_length", truncation=True,
                         max_length=77, return_tensors="pt").input_ids.to(device)
    cond_emb = text_encoder(cond_ids)[0]

    # uncond embeddings for CFG
    uncond_ids = tokenizer([""] * n, padding="max_length",
                           truncation=True, max_length=77,
                           return_tensors="pt").input_ids.to(device)
    uncond_emb = text_encoder(uncond_ids)[0]

    # --- DDIM 取樣 ---
    latents = torch.randn(n, 4, 32, 32, device=device)
    scheduler.set_timesteps(NUM_STEPS, device=device)

    # --- 逐步去噪（Classifier-Free Guidance）---
    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)                        # concat cond + uncond，進行雙路預測
        encoder_hidden = torch.cat([uncond_emb, cond_emb])

        noise_pred = unet(latent_model_input, t,
                          encoder_hidden_states=encoder_hidden).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + GUIDANCE * (noise_cond - noise_uncond)   # CFG 推理核心：強化 prompt 對結果的影響力

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # --- 使用 VAE 解碼 latent 成圖 ---
    latents = latents / 0.18215
    imgs = vae.decode(latents).sample         # [-1,1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2        # → [0,1]
    return imgs

def main():
    # --- 載入 CLIP & VAE  ---
    clip_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_name).to(device).eval()

    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device).eval()
    vae.requires_grad_(False)

    # --- 載入 UNet ---
    unet = UNet2DConditionModel(**unet_cfg).to(device)
    unet.load_state_dict(torch.load(UNET_CKPT, map_location="cpu"))
    unet.eval()

    # --- Scheduler ---
    # scheduler = DDPMScheduler()   # 舊版
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",  
    )
    scheduler.set_timesteps(NUM_STEPS, device=device)

    # --- 讀取 test.json ---
    with open("../test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    ids    = list(test_data.keys())
    prompts = [test_data[k]["text_prompt"] for k in ids]
    fnames  = [test_data[k]["image_name"]  for k in ids]

    # === 批次生成並儲存圖片 ===
    to_pil = transforms.ToPILImage()
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        batch_fnames  = fnames[i:i+BATCH_SIZE]

        imgs = sample_batch(batch_prompts, unet, vae,
                            tokenizer, text_encoder, scheduler, device)

        for img_tensor, fname in zip(imgs, batch_fnames):
            to_pil(img_tensor.cpu()).save(SAVE_DIR / fname)

    print("All images saved to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()