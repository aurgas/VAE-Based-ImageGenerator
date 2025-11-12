# VAE-Based-ImageGenerator
This project uses a Variational auto encoder to generate images. The key idea here is to increase computational efficiency by skipping the auto-encoder training by using transfer-learning techniques. We use pre trained resnet18, by removing the last classification layer we get a latent vector that we feed into a decoder trained on a small dataset.
