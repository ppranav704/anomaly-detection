import numpy as np

def detect_anomalies(reconstruction_errors, threshold):
    anomalies_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
    return anomalies_indices

def load_reconstruction_errors(file_path):
    with open(file_path, 'r') as f:
        reconstruction_errors = [float(line.strip()) for line in f]
    return reconstruction_errors

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    return sentences

def main():
    # File paths
    errors_file = r'\anomaly_detection\src\pipeline\reconstruction_errors.txt'
    messages_file = r'\anomaly_detection\data\messages.txt'

    # Load reconstruction errors and sentences
    reconstruction_errors = load_reconstruction_errors(errors_file)
    sentences = load_sentences(messages_file)

    # Set a threshold for anomaly detection
    threshold = 1375

    # Detect anomalies
    anomaly_indices = detect_anomalies(reconstruction_errors, threshold)

    # Print lengths and content for debugging
    print("Length of reconstruction_errors:", len(reconstruction_errors))
    print("Length of sentences:", len(sentences))
    print("Anomaly indices:", anomaly_indices)

    # Ensure anomaly indices are within range of sentences
    anomaly_indices = [idx for idx in anomaly_indices if idx < len(sentences)]

    if anomaly_indices:
        # Get original messages of detected anomalies
        detected_anomalies = [sentences[i] for i in anomaly_indices]

        # Print detected anomalies
        print("Detected Anomalies:")
        for anomaly in detected_anomalies:
            print(anomaly)
    else:
        print("No anomalies detected.")

    # Visualize anomalies or take further actions as needed

if __name__ == "__main__":
    main()
