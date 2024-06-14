import time
import torch
import torch.nn as nn
import numpy as np
import fasttext
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
import aiofiles
import logging
import os

# Initialize the FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    "fasttext_model_path": "src/pipeline/fasttext_model.bin",
    "inference_info_path": "src/pipeline/inference_info2.pth",
    "model_path": "src/pipeline/model.pth",
    "data_file_path": "/data/messages.txt",
    "anomaly_threshold": 1375
}

# Variational Autoencoder model definition
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_size, 256)
        self.norm1 = nn.LayerNorm(256, eps=1e-12, elementwise_affine=True)
        self.fc21 = nn.Linear(256, latent_size)
        self.fc22 = nn.Linear(256, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, 256)
        self.norm2 = nn.LayerNorm(256, eps=1e-12, elementwise_affine=True)
        self.fc4 = nn.Linear(256, input_size)

        # PReLU activations
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h1 = self.norm1(self.prelu1(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.norm2(self.prelu2(self.fc3(z)))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function for VAE with MSE
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# Load pre-trained FastText model
model_fasttext = fasttext.load_model(CONFIG["fasttext_model_path"])

# Load inference information
inference_info = torch.load(CONFIG["inference_info_path"])
scaler_info = inference_info['scaler']
input_size = inference_info['input_size']
latent_size = inference_info['latent_size']

# Load VAE model
model_vae = VAE(input_size, latent_size)
model_vae.load_state_dict(torch.load(CONFIG["model_path"]))
model_vae.eval()

# Standardize the data using the previously saved scaler
scaler = StandardScaler()
scaler.partial_fit(scaler_info)

# Load sentences from a text file
async def load_sentences(file_path):
    sentences = []
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        async for line in file:
            sentences.append(line.strip())
    return sentences

# Convert sentences to vectors using FastText
def get_sentence_vector(sentences, model_fasttext):
    return np.array([model_fasttext.get_sentence_vector(sentence) for sentence in sentences])

@app.get("/live_predict/")
async def live_predict():
    try:
        sentences = await load_sentences(CONFIG["data_file_path"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data file not found")

    log_vectors = get_sentence_vector(sentences, model_fasttext)
    x_data = scaler.transform(log_vectors)
    x_inference_tensor = torch.tensor(x_data, dtype=torch.float32)

    anomalies = []
    non_anomalies = []
    anomaly_count = 0
    non_anomaly_count = 0

    for sentence, log_vector in zip(sentences, x_inference_tensor):
        try:
            start_time = time.time()
            x_inference_tensor = torch.tensor([log_vector], dtype=torch.float32)
            with torch.no_grad():
                recon_sample, mu, logvar = model_vae(x_inference_tensor)
                reconstruction_error = loss_function(recon_sample, x_inference_tensor, mu, logvar).item()

            is_anomaly = reconstruction_error > CONFIG["anomaly_threshold"]
            result = {
                "sentence": sentence,
                "reconstruction_error": reconstruction_error,
                "anomaly": is_anomaly,
                "execution_time": time.time() - start_time
            }

            if is_anomaly:
                anomalies.append(result)
                anomaly_count += 1
            else:
                non_anomalies.append(result)
                non_anomaly_count += 1
        except Exception as e:
            logger.error(f"Error processing sentence: {sentence}\nError: {e}")

    response = {
        "anomalies": anomalies,
        "non_anomalies": non_anomalies,
        "anomaly_count": anomaly_count,
        "non_anomaly_count": non_anomaly_count
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9030)
