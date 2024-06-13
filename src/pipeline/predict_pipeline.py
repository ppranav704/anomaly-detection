import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import fasttext
import seaborn as sns
import matplotlib.pyplot as plt


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
model_fasttext = fasttext.load_model(r"\anomaly_detection\src\pipeline\model_messages.bin")

# Load sentences from a text file
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file]
    return sentences

# Convert the sentences to vectors using FastText
def get_sentence_vector(sentence):
    return model_fasttext.get_sentence_vector(sentence)

# Load inference information
inference_info = torch.load(r'\anomaly_detection\src\pipeline\inference_info.pth')  
scaler_info = inference_info['scaler']
input_size = inference_info['input_size']
latent_size = inference_info['latent_size']

# Example: Load sentences from a text file
file_path = r'\anomaly_detection\data\messages.txt'  
sentences = load_sentences(file_path)

# Convert the sentences to vectors using FastText
log_vectors = np.array([get_sentence_vector(sentence) for sentence in sentences])

# Standardize the data using the previously saved scaler
scaler = StandardScaler()
x_data = scaler.fit_transform(log_vectors)

# Convert the NumPy array to PyTorch tensor
x_inference_tensor = torch.tensor(x_data, dtype=torch.float32)

# Set batch size
batch_size = 64

# Create DataLoader for inference
inference_dataset = TensorDataset(x_inference_tensor)
inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model_vae = VAE(input_size=input_size, latent_size=latent_size)
model_vae.load_state_dict(torch.load(r'\anomaly_detection\src\pipeline\model.pth'))

# Lists to store reconstruction errors
reconstruction_errors = []

# Perform inference
model_vae.eval()
with torch.no_grad():
    for batch in inference_dataloader:
        x_batch = batch[0]
        for x_sample in x_batch:
            x_sample = x_sample.unsqueeze(0)  # Add batch dimension for single sample
            recon_sample, mu, logvar = model_vae(x_sample)
            loss = loss_function(recon_sample, x_sample, mu, logvar)
            reconstruction_errors.append(loss.item())

# Save the reconstruction errors to a text file
with open(r'\anomaly_detection\src\pipeline\reconstruction_errors.txt', 'w') as f:
    for error in reconstruction_errors:
        f.write(str(error) + '\n')
        
# Print and plot reconstruction errors
print("Inference Reconstruction Errors:")
print(reconstruction_errors)

# Plot histogram with KDE for filtered reconstruction errors
sns.histplot(reconstruction_errors, bins=50, kde=True, kde_kws={'bw_method': 0.2})  
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Inference Reconstruction Error Distribution")
plt.show()








