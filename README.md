# Anomaly Detection on Unstructured Streaming Logs using Variational Autoencoder

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Scripts Description](#scripts-description)
  - [Preprocessing](#preprocessing)
  - [Prediction Pipeline](#prediction-pipeline)
  - [Anomaly Detection](#anomaly-detection)
- [FastAPI Application](#fastapi-application)
- [Requirements](#requirements)
- [License](#license)

## Introduction
This project implements an anomaly detection system for unstructured streaming logs using a Variational Autoencoder (VAE). The system preprocesses the raw logs to generate embeddings using FastText, and then detects anomalies based on reconstruction errors.

## Project Structure
ANOMALY_DETECTION
├── dlproject.egg-info/
├── data/
├── src/
│   ├── pipeline/
│   │   ├── anomaly_detection/
│   │   ├── __pycache__/
│   │   ├── anomaly_detection.py
│   │   ├── app.py
│   │   ├── fasttext_model.bin
│   │   ├── inference_info.pth
│   │   ├── inference_info_bert.pth
│   │   ├── model.pth
│   │   ├── modelbert.pth
│   │   ├── predict_pipeline.py
│   │   ├── preprocessing.py
│   │   ├── pretrained_bert_embeddings.npy
│   │   ├── reconstruction_errors.txt
│   │   ├── validation_set.pth
│   │   ├── validation_set_bert.pth
│   │   ├── __init__.py
│   ├── app1.py
│   ├── exception.py
│   ├── logger.py
│   ├── __init__.py
├── venv/
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py




## Setup and Installation

1. **Clone the repository:**
   
   git clone https://github.com/yourusername/dlproject.git
   cd dlproject


2. Create a virtual environment:

   'conda create -p venv python==3.10 -y'

3. Activate the virtual environment:

   'conda activate \anomaly_detection\venv'

4. Install the required packages:

   'pip install -r requirements.txt'


## Usage

1. Preprocessing
   Run the preprocessing.py script to convert raw data into FastText embeddings using the CBOW model.

   'python src/pipeline/preprocessing.py'

2. Prediction Pipeline
   Run the predict_pipeline.py script to generate results and store reconstruction errors.

   'python src/pipeline/predict_pipeline.py'

3. Anomaly Detection
   Run the anomaly_detection.py script to detect anomalies based on the set threshold.

   'python src/pipeline/anomaly_detection.py'

## FastAPI Application
   The app1.py script in the src directory is a FastAPI application for single message anomaly detection.

   To run the FastAPI application:

   'uvicorn src.app1:app --reload'

## Scripts Description
Preprocessing
 'preprocessing.py':

    - Converts raw data into FastText embeddings using the CBOW model.
    - Stores the embeddings for further processing.

Prediction Pipeline
 'predict_pipeline.py':

    - Generates results and stores reconstruction errors in a text file (reconstruction_errors.txt).

Anomaly Detection
 'anomaly_detection.py':

    - Prints detected anomalies based on a set threshold of reconstruction errors.

## Requirements
All required Python packages are listed in requirements.txt. Install them using:

 'pip install -r requirements.txt'

## License
This project is licensed under the MIT License. See the LICENSE file for details.

   