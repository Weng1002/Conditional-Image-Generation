import argparse, os, glob, json, math, random, itertools, time
from pathlib import Path
from typing import List

import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import lpips

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ------------------------- 自定義資料集 ------------------------- #
class TextImageDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 caption_file: str,
                 tokenizer: CLIPTokenizer,
                 size: int = 256,
                 caption_dropout_prob: float = .1):
        self.img_paths: List[str] = glob.glob(os.path.join(img_dir, "*.png"))
        assert len(self.img_paths), f"No png in {img_dir}"
        with open(caption_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        self.tokenizer = tokenizer
        self.size = size
        self.caption_dropout_prob = caption_dropout_prob

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC), 
            transforms.RandomHorizontalFlip(),           # 隨機水平翻轉
            transforms.ToTensor(),                       # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])        # 正規化為[-1,1]
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        key = Path(img_path).stem                # e.g.,  "0_attack1_3"
        key = "_".join(key.split("_")[:-1])      # → "0_attack1"
        info = self.captions[key]

        # 90 % 使用內容；10 % 空字串 (uncond)  –––→  CFG 訓練技巧
        if random.random() < self.caption_dropout_prob:
            caption = ""
        else:
            cap = random.choice(info["given_description"])
            action = info["action_description"]
            caption = cap if action == "" else f"{cap} {action}"

        token = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        ).input_ids[0]

        return {"pixel_values": img, "input_ids": token}


# ------------------------ 預覽生圖 (CFG) ------------------------ #
@torch.no_grad()
def preview_samples(unet, vae, text_encoder, tokenizer, device, step,
                    save_dir: Path,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 6.0):
    unet.eval()
    scheduler = DDPMScheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)

    prompts = [
        "A red tree monster with a skull face and twisted branches.",
        "Blood-toothed monster with spiked fur, wielding an axe, moving fiercely.",
        "Gray vulture monster with wings, sharp beak, and trident.",
        "Small purple fish-like creature with one giant eye and pink fins, being hit."
    ]

    save_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        
        # -------- cond / uncond embeddings -------- #
        text_ids = tokenizer(prompt, return_tensors="pt",
                             padding="max_length",
                             truncation=True,
                             max_length=77).input_ids.to(device)
        cond_emb = text_encoder(text_ids)[0]

        uncond_ids = tokenizer("", return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=77).input_ids.to(device)
        uncond_emb = text_encoder(uncond_ids)[0]

        # latent 初始化為高斯噪聲
        latent = torch.randn(1, 4, 32, 32, device=device)

        # 跑遍所有 scheduler 的 timestep，逐步反推生成圖片
        for t in scheduler.timesteps:
            latent_in = torch.cat([latent] * 2)
            emb = torch.cat([uncond_emb, cond_emb])

            noise_pred = unet(latent_in, t, encoder_hidden_states=emb).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latent = scheduler.step(noise_pred, t, latent).prev_sample

        # 最後用 VAE decoder 將 latent 轉回圖片
        latent = latent / 0.18215
        img = vae.decode(latent).sample
        img = (img.clamp(-1, 1) + 1) / 2
        img = transforms.ToPILImage()(img[0].cpu())
        img.save(save_dir / f"step{step:07d}_{i}.png")

    unet.train()


# ------------------------- 主要訓練程式 ------------------------- #
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LPIPS 損失函數 
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    
    # ---------- 預訓練元件 (規定可用) ---------- #
    clip_name = "openai/clip-vit-base-patch32"
    vae_name  = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_name).to(device).eval()
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)

    # ---------- 建立 UNet ---------- #
    base_channels = args.base_channels       
    unet = UNet2DConditionModel(
        sample_size      = 32,
        in_channels      = 4,
        out_channels     = 4,
        cross_attention_dim = 512,
        layers_per_block = 2,
        block_out_channels = (
            base_channels,
            int(base_channels * 1.5),
            base_channels * 2,
            base_channels * 3,
        ),
        down_block_types = (
            "CrossAttnDownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types = (
            "CrossAttnUpBlock2D", "UpBlock2D",
            "CrossAttnUpBlock2D", "UpBlock2D",
        ),
    ).to(device)
    unet.enable_gradient_checkpointing()

    # === 如果有指定 resume checkpoint 就載入 ===
    start_epoch = 0
    if args.resume:
        print(f"▶️ Resume from checkpoint: {args.resume}")
        unet.load_state_dict(torch.load(args.resume, map_location="cpu"))
        try:
            start_epoch = int(Path(args.resume).parent.name.replace("unet_Epoch", ""))
        except:
            print("Resume filename format not matched, will continue from epoch 0")

    # ---------- Scheduler & Opt ---------- #
    noise_scheduler = DDPMScheduler()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    # ---------- 資料集 ---------- #
    train_set = TextImageDataset(
        img_dir="../train",
        caption_file="../train_info.json",
        tokenizer=tokenizer,
        caption_dropout_prob=0.1
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_per_device,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Dataset: {len(train_set)} images  —  {len(train_loader)} steps/epoch")

    # ---------- 輸出資料夾 ---------- #
    ckpt_dir   = Path("outputs/ckpt"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path("outputs/samples_img"); sample_dir.mkdir(parents=True, exist_ok=True)

    # ------------------- 訓練迴圈 ------------------- #
    unet.train()
    global_step = 0 
    
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            global_step += 1
            with torch.cuda.amp.autocast():
                
                # -- encode text / images --
                enc_hidden = text_encoder(batch["input_ids"].to(device))[0]

                latents_dist = vae.encode(batch["pixel_values"].to(device)).latent_dist     # 文字 + 圖片 → 編碼後加噪聲
                latents = latents_dist.sample() * 0.18215                                   # scaling 係數
                kl_loss = 0.01 * latents_dist.kl().mean()                                   # KL divergence loss

                noise      = torch.randn_like(latents)                                      
                timesteps  = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.size(0),), device=device, dtype=torch.long
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=enc_hidden).sample   # 預測 noise

                # === MSE loss ===
                mse_loss = F.mse_loss(noise_pred, noise, reduction="mean")

                # === LPIPS loss ===
                with torch.no_grad():
                    denoised_latents = noisy_latents - noise_pred                            # MSE + LPIPS + KL → 總 loss
                    recon = vae.decode(denoised_latents / 0.18215).sample                    # [-1, 1]
                    target = batch["pixel_values"].to(device)
                lpips_loss = lpips_fn(recon, target).mean()

                # === Final loss ===
                loss = (mse_loss + 0.1 * lpips_loss + kl_loss) / args.accum_steps

            scaler.scale(loss).backward()                                                     # 反向傳播與梯度累積

            # -- Gradient accumulation --
            if global_step % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.zero_grad(set_to_none=True)

            # -- 日誌 --
            pbar.set_description(f"E{epoch+1} G{global_step} loss {loss.item()*args.accum_steps:.4f}")

            # -- 預覽圖 --
            if global_step % args.preview_freq == 0:
                preview_samples(unet, vae, text_encoder, tokenizer,
                                device, global_step, sample_dir,
                                guidance_scale=args.cfg_scale)

        # === 每個 epoch 結束後儲存目前模型 ===
        epoch_ckpt = ckpt_dir / f"unet_Epoch{epoch+1:03d}"
        epoch_ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(unet.state_dict(), epoch_ckpt / "pytorch_model.bin")
        print(f"✓ Epoch {epoch+1} checkpoint saved to {epoch_ckpt}")

    #  最後再存一次
    final_path = ckpt_dir / f"unet_final"
    final_path.mkdir(exist_ok=True)
    torch.save(unet.state_dict(), final_path / "pytorch_model.bin")
    print("🎉 Training finished & final checkpoint saved.")


# ------------------------- CLI 參數 ------------------------- #
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",            type=int, default=300)
    p.add_argument("--batch_per_device",  type=int, default=64)
    p.add_argument("--accum_steps",       type=int, default=2,  help="effective batch = batch_per_device × accum_steps")
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--base_channels",     type=int, default=256, help="UNet first-level channels")
    p.add_argument("--preview_freq",      type=int, default=500)     # steps
    p.add_argument("--save_freq",         type=int, default=5_000)   # steps
    p.add_argument("--cfg_scale",         type=float, default=6.0)
    p.add_argument("--resume",            type=str,   default=None)
    p.add_argument("--seed",              type=int,   default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
