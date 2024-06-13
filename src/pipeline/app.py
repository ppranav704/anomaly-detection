from fastapi import FastAPI
import time
import torch
import torch.nn as nn
import numpy as np
import fasttext
from sklearn.preprocessing import StandardScaler
import aiofiles

app = FastAPI()

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

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_divergence

# Load pre-trained FastText model
model_fasttext = fasttext.load_model(r"\anomaly_detection\src\pipeline\fasttext_model.bin")

# Load inference information
inference_info = torch.load(r'\anomaly_detection\src\pipeline\inference_info.pth')  
scaler_info = inference_info['scaler']
input_size = inference_info['input_size']
latent_size = inference_info['latent_size']

# Load sentences from a text file
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]
    return sentences

# Convert the sentences to vectors using FastText
def get_sentence_vector(sentences, model_fasttext):
    return np.array([model_fasttext.get_sentence_vector(sentence) for sentence in sentences])

# Example: Load sentences from a text file
file_path = r'\anomaly_detection\data\messages.txt'  
sentences = load_sentences(file_path)

# Convert the sentences to vectors using FastText
log_vectors = get_sentence_vector(sentences, model_fasttext)

# Standardize the data using the previously saved scaler
scaler = StandardScaler()
scaler.partial_fit(log_vectors)

# Transform the data using the fitted scaler
x_data = scaler.transform(log_vectors)

# Convert the NumPy array to PyTorch tensor
x_inference_tensor = torch.tensor(x_data, dtype=torch.float32)

# Load VAE model
model_vae = VAE(input_size, latent_size)
model_vae.load_state_dict(torch.load(r'\anomaly_detection\src\pipeline\model.pth'))
model_vae.eval()

@app.get("/live_predict/")
async def live_predict():
    anomalies = []  # List to store anomaly results
    non_anomalies = []  # List to store non-anomaly results
    anomaly_count = 0
    non_anomaly_count = 0

    async with aiofiles.open(r"\anomaly_detection\data\messages.txt", "r") as file:
        async for line in file:
            sentence = line.strip()

            try:
                start_time = time.time()  # Record start time

                # Convert sentence to vector using FastText
                log_vector = model_fasttext.get_sentence_vector(sentence)

                # Transform the data using the fitted scaler
                x_data = scaler.transform(np.array([log_vector]))

                # Convert to PyTorch tensor
                x_inference_tensor = torch.tensor(x_data, dtype=torch.float32)

                # Perform inference
                model_vae.eval()
                with torch.no_grad():
                    recon_sample, mu, logvar = model_vae(x_inference_tensor)
                    reconstruction_error = loss_function(recon_sample, x_inference_tensor, mu, logvar).item()

                # Check if reconstruction error exceeds threshold
                threshold = 1375
                is_anomaly = reconstruction_error > threshold

                # Create result dictionary
                result = {
                    "sentence": sentence,
                    "reconstruction_error": reconstruction_error,
                    "anomaly": is_anomaly,
                    "execution_time": time.time() - start_time  # Calculate execution time
                }

                # Append result to the appropriate list
                if is_anomaly:
                    anomalies.append(result)
                    anomaly_count += 1
                else:
                    non_anomalies.append(result)
                    non_anomaly_count += 1

            except Exception as e:
                print(f"Error processing sentence: {sentence}\nError: {e}")

    # Print anomalies and non-anomalies
    print("Anomalies:")
    for anomaly in anomalies:
        print(anomaly)

    print("Non-Anomalies:")
    for non_anomaly in non_anomalies:
        print(non_anomaly)

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

