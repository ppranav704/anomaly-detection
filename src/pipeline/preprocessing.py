import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import fasttext

def train_fasttext_model(input_file, output_model_path, dim=300, minn=2, maxn=5, minCount=1,  model='cbow'):
    """
    Train a FastText model on the input file and save it to the output_model_path.
    
    Parameters:
        input_file (str): Path to the input file containing text data.
        output_model_path (str): Path to save the trained FastText model.
        dim (int): Dimensionality of word vectors (default is 300).
        minn (int): Minimum length of char n-grams (default is 2).
        maxn (int): Maximum length of char n-grams (default is 5).
        model (str): Model architecture ('skipgram' or 'cbow') (default is 'cbow').
    """
    model = fasttext.train_unsupervised(input_file, minn=minn, maxn=maxn, dim=dim, minCount=minCount, model=model)
    model.save_model(output_model_path)
    
    #print(model.words)
    
    # Print the dimensions of the saved model
    #print("Dimensions of the saved model:", model.get_dimension())
    
if __name__ == "__main__":
    input_file = r'\anomaly_detection\data\messages1.txt'
    output_model_path = r'\anomaly_detection\src\pipeline\model_messages.bin'
    train_fasttext_model(input_file, output_model_path)
