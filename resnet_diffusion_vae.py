# resnet_cond_diffusion.py
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DConditionModel, DDPMScheduler
from resnet_vae import ResNetVAE  # your ResNet encoder / VAE file (must define decode/encode as needed)

# ---------- Config ----------
CHECKPOINT_PATH = "/Users/poulam/Desktop/Generative Bullshit/vae_weights.pth"  # if you want to load ResNetVAE weights
UNET_PRETRAINED = "runwayml/stable-diffusion-v1-5"   # pretrained stable-diffusion model id for weights (we'll take the UNet)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = "/Users/poulam/Desktop/Generative Bullshit/diffusion_from_resnet.png"

# conditioning projection settings (match UNet cross-attention dim)
CROSS_ATTENTION_DIM = 768   # stable-diffusion v1.x UNet uses 768 cross-attention dim
TEXT_SEQ_LEN = 77          # CLIP token length SD uses by default (we'll mimic this)
LATENT_DIM = 256           # your ResNet latent dim (must match your trained VAE if loading weights)
# ----------------------------

# Small helper: project your ResNet latent z -> shape (batch, TEXT_SEQ_LEN, CROSS_ATTENTION_DIM)
class ZToCrossAttn(nn.Module):
    def __init__(self, latent_dim, seq_len=TEXT_SEQ_LEN, cross_dim=CROSS_ATTENTION_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.cross_dim = cross_dim
        # project latent vector to seq_len * cross_dim, then reshape
        self.proj = nn.Linear(latent_dim, seq_len * cross_dim)

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.proj(z)               # (batch, seq_len * cross_dim)
        x = x.view(z.size(0), self.seq_len, self.cross_dim)
        return x

# Load UNet from diffusers (only UNet, not full pipeline)
print("Loading UNet from pretrained checkpoint (may download ~500MB)...")
unet = UNet2DConditionModel.from_pretrained(UNET_PRETRAINED, subfolder="unet").to(DEVICE)
unet.eval()

# Scheduler (DDPM) - you can switch to DDIM/PNDM as preferred
scheduler = DDPMScheduler.from_pretrained(UNET_PRETRAINED, subfolder="scheduler")

# Load your ResNetVAE encoder (we assume it exposes encode() producing z of size LATENT_DIM)
print("Loading your ResNetVAE encoder (from resnet_vae.py)...")
resnet_vae = ResNetVAE(latent_dim=LATENT_DIM).to(DEVICE)
# optionally load checkpoint if available and matches arch
try:
    ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # handle both state_dict or dict with model_state
    if isinstance(ck, dict) and "model_state" in ck:
        resnet_vae.load_state_dict(ck["model_state"])
    else:
        resnet_vae.load_state_dict(ck)
    print("Loaded ResNetVAE weights from", CHECKPOINT_PATH)
except Exception as e:
    print("Could not load ResNetVAE checkpoint (continuing with randomly initialized encoder). Error:", e)

resnet_vae.eval()

# create adapter to project z -> cross-attn
z_to_attn = ZToCrossAttn(latent_dim=LATENT_DIM).to(DEVICE)

# ---------- Generation routine ----------
@torch.no_grad()
def generate_from_resnet(image_path: str = None, num_steps: int = 50, batch_size=1):
    """
    If image_path provided: encode that image to z and condition generation on it (reconstruction / conditional).
    If image_path is None: sample z ~ N(0, I) and generate images conditioned on that z.
    """
    # if image_path: produce z by encoding an image (you may prefer to use mu instead of sampling)
    if image_path:
        img = Image.open(image_path).convert("RGB")
        tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])  # match your encoder's training size
        x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # assume encode returns (recon, mu, logvar) or (z, mu, logvar) depending on your resnet_vae implementation
            # Here we try both styles safely:
            enc_out = resnet_vae.encoder(x) if hasattr(resnet_vae, "encoder") else None
            try:
                # if your ResNetVAE has encode method returning (z, mu, logvar)
                z, mu, logvar = resnet_vae.encode(x)
            except Exception:
                # fallback: do forward then take mu
                _, mu, logvar = resnet_vae.forward(x)
                z = mu  # use mean for deterministic conditioning
        z = z.to(DEVICE)
    else:
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)

    # project to cross-attention embeddings
    encoder_hidden_states = z_to_attn(z)  # shape (batch, seq_len, cross_dim)

    # Prepare starting noise for UNet. UNet expects latents shape: (B, C, H, W)
    # For pixel-space UNet in SD v1.5 we will generate 64x64 or 128x128 depending on UNet configuration.
    # We'll query unet.config.sample_size to pick size:
    sample_size = getattr(unet.config, "sample_size", 64)  # fallback 64
    channels = unet.config.in_channels if hasattr(unet.config, "in_channels") else 3

    # Start with pure Gaussian noise in pixel-space
    x = torch.randn(batch_size, channels, sample_size, sample_size, device=DEVICE)

    # prepare timesteps (descending)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(DEVICE)

    for t in timesteps:
        # UNet forward expects: sample, timestep (int tensor), encoder_hidden_states
        # but diffusers UNet expects timesteps as tensor same dtype device
        t_tensor = torch.tensor([int(t.item())]*batch_size, device=DEVICE)
        # predict noise / denoised image
        model_pred = unet(x, t_tensor, encoder_hidden_states=encoder_hidden_states).sample
        # step through scheduler
        step_out = scheduler.step(model_pred, t, x)
        x = step_out.prev_sample

    # clip and convert to [0..1]
    x = (x - x.min()) / (x.max() - x.min())
    return x

# ---------- Run generation ----------
if __name__ == "__main__":
    # 1) conditional generation from an image (comment out if you want unconditional)
    # generated = generate_from_resnet(image_path="/path/to/your/conditioning_image.jpg", num_steps=50, batch_size=1)

    # 2) unconditional sampling from random z
    generated = generate_from_resnet(image_path=None, num_steps=50, batch_size=1)

    # Save first sample
    save_image(generated[0].cpu(), OUT_PATH)
    print("Saved generated image to:", OUT_PATH)