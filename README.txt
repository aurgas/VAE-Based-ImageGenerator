# 🧠 ResNet-Diffusion VAE  
### Hybrid Latent Image Generator using ResNet Encoder + Diffusion Decoder

This project implements a **hybrid generative model** that combines a **pretrained ResNet encoder** with a **diffusion-based decoder** (UNet architecture) for image generation.  
Inspired by *Stable Diffusion*, this design replaces the traditional VAE decoder with a pretrained **denoising diffusion model**, allowing for sharper and more diverse image synthesis.

---

## 🚀 Features

- 🧩 **ResNet Encoder** – Pretrained ResNet18/34/50 backbone for feature extraction.  
- 🌫️ **Diffusion Decoder** – Pretrained UNet (from 🤗 Hugging Face Diffusers) as the denoising decoder.  
- 🧠 **Variational Latent Space** – VAE-style latent sampling using mean & variance projections.  
- 🔄 **Conditional Generation** – Image generation conditioned on learned latent embeddings.  
- ⚡ **Supports both pixel-space and latent-space diffusion**  
- 💾 **Modular Design** – Easy to swap in different backbones or diffusion models.

---

## 🧱 Project Structure

ResNet-Diffusion-VAE/
│
├── resnet_vae.py               # ResNet-based VAE model (ResNet encoder + transpose decoder)
├── resnet_diffusion_vae.py     # Hybrid model using ResNet encoder + diffusion decoder
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Required Python libraries
├── vae_weights.pth             # (optional) Trained VAE checkpoint
└── generated_sample.png        # Example generated output

---

## 🧩 Model Architecture

### 1. Encoder  
- Uses **ResNet18** pretrained on ImageNet (`torchvision.models.resnet18`)  
- Extracts high-level features → outputs mean (`μ`) and log variance (`logσ²`) vectors  
- Produces a latent embedding `z = μ + σ ⊙ ε`

### 2. Diffusion Decoder  
- Based on a pretrained **Stable Diffusion UNet** (`UNet2DConditionModel` from `diffusers`)  
- Denoises Gaussian noise into a final image, conditioned on `z`  
- Uses a small adapter MLP to project `z` into the UNet’s **cross-attention space (768-dim)**

---

## 🧠 How It Works

1. **Encode an Image**  
   - Input → ResNet Encoder → latent vector `z` (256-dim default)  
2. **Condition the Diffusion Decoder**  
   - Project `z` → 768-dim cross-attention embedding  
3. **Generate Image**  
   - Start from random noise → Diffusion UNet denoises over multiple timesteps  
   - Final output: synthetic image consistent with encoded latent distribution  

---

## 🧪 Example Usage

### Generate a Random Image
```bash
python resnet_diffusion_vae.py
```

### Conditional Generation (from image)
In `resnet_diffusion_vae.py`, uncomment:
```python
generated = generate_from_resnet(image_path="path/to/image.jpg", num_steps=50)
```

This will encode the given image → latent `z` → condition diffusion model → reconstruct a new variant.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/ResNet-Diffusion-VAE.git
cd ResNet-Diffusion-VAE

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧩 Requirements

See [`requirements.txt`](./requirements.txt)

Main libraries:
- torch, torchvision
- diffusers
- transformers
- accelerate
- safetensors
- Pillow
- tqdm

---

## 🧰 Notes

- The first run will **download pretrained diffusion weights (~3.4 GB)** from Hugging Face.  
- Once downloaded, they’re cached in `~/.cache/huggingface/`.  
- Works with both CPU and CUDA (GPU recommended).  
- To reduce VRAM usage, enable half-precision (`torch.float16`) inference.

---

## 📈 Future Improvements

- [ ] Add VAE fine-tuning with diffusion guidance  
- [ ] Integrate Stable Diffusion VAE latent-space decoder  
- [ ] Add ControlNet or LoRA conditioning adapters  
- [ ] Support for text + image joint conditioning  

---

## 🧑‍💻 Author

**Poulam Saha**  
Generative AI & Deep Learning Enthusiast  
📍 India  
🌐 *GitHub*: [github.com/<your-username>](https://github.com/<your-username>)  

---

## 🪪 License

MIT License © 2025 Poulam Saha
