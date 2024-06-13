import numpy as np

def detect_anomalies(reconstruction_errors, threshold):
    anomalies_indices = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
    return anomalies_indices

def main():
    # Load reconstruction errors from the stored file
    with open(r'D:\Projects\dlproject\src\pipeline\reconstruction_errors.txt', 'r') as f:
        reconstruction_errors = [float(line.strip()) for line in f]

    # Load sentences from the stored file
    with open(r'D:\Projects\dlproject\notebook\data\messages.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]

    # Set a threshold for anomaly detection
    threshold = 365.04

    # Detect anomalies
    anomaly_indices = detect_anomalies(reconstruction_errors, threshold)

    # Print lengths and content for debugging
    print("Length of reconstruction_errors:", len(reconstruction_errors))
    print("Length of sentences:", len(sentences))
    print("Anomaly indices:", anomaly_indices)

    # Ensure anomaly indices are within range
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









