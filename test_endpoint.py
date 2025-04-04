import boto3
import json
import numpy as np
from PIL import Image

# Replace with your endpoint name and region
ENDPOINT_NAME = 'cifar10-endpoint'
REGION = 'us-east-1'

# Create a SageMaker runtime client
client = boto3.client('sagemaker-runtime', region_name=REGION)

def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    # Resize to 32x32 pixels
    img = img.resize((32, 32))
    # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Normalize using CIFAR-10 stats
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_array = (img_array - mean) / std
    # Ensure the shape is (32, 32, 3) and then convert to list for JSON serialization
    return img_array.tolist()

# Provide the path to a valid test image
test_image_path = "0002.jpg"  # Update this path
payload_data = preprocess_image(test_image_path)
payload = json.dumps({"instances": [payload_data]})

print("Payload being sent:")
print(payload)

# Invoke the endpoint
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read().decode())
print("Inference result:", result)
