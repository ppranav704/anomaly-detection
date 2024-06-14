import time
import torch
import torch.nn as nn
import numpy as np
import fasttext
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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
    "inference_info_path": "src/pipeline/inference_info.pth", 
    "model_path": "src/pipeline/model.pth",
    "data_file_path": "/data/messages1.txt",
    "anomaly_threshold": 62.81
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = self.norm2(self.prelu2(self.fc3(z)))
        return self.fc4(h3)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function for VAE        
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# Load pre-trained FastText model
model_fasttext = fasttext.load_model(CONFIG["fasttext_model_path"])

# Load inference information 
inference_info = torch.load(CONFIG["inference_info_path"])
input_size = inference_info['input_size'] 
latent_size = inference_info['latent_size']
scaler_info = inference_info['scaler']

# Initialize StandardScaler
scaler = StandardScaler()  
scaler.partial_fit(scaler_info)

# Load VAE model
model_vae = VAE(input_size, latent_size)
model_vae.load_state_dict(torch.load(CONFIG["model_path"]))
model_vae.eval()

# WebSocket manager for handling connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# WebSocket endpoint for real-time predictions
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            sentences = data.split("\n")
            for sentence in sentences:
                result = await process_sentence(sentence)
                await manager.broadcast(result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Process a single sentence for prediction  
async def process_sentence(sentence: str):
    try:
        start_time = time.time()
        
        # Convert sentence to vector using FastText
        log_vector = model_fasttext.get_sentence_vector(sentence)
        x_data = scaler.transform(np.array([log_vector]))
        x_inference_tensor = torch.tensor(x_data, dtype=torch.float32)

        # Perform inference with VAE
        with torch.no_grad():
            recon_sample, mu, logvar = model_vae(x_inference_tensor)
            reconstruction_error = loss_function(recon_sample, x_inference_tensor, mu, logvar).item()
        
        is_anomaly = reconstruction_error > CONFIG["anomaly_threshold"]
        execution_time = time.time() - start_time
        
        response = {
            "sentence": sentence, 
            "reconstruction_error": reconstruction_error,
            "anomaly": is_anomaly,
            "execution_time": execution_time
        }
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sentence: {sentence}\nError: {e}")

# HTTP endpoint to receive streaming data  
@app.post("/streaming-endpoint")
async def receive_streaming_data(data: str):
    sentences = data.split("\n")
    results = []
    for sentence in sentences:
        result = await process_sentence(sentence)
        results.append(result)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9020)