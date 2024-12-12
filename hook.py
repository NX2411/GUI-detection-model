# Import your Libraries 
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba_array 

# Define hook function
def hook_fn(module, input, output):
    intermediate_features.append(output)

# Define feature extraction function
def extract_features(model, img, layer_index=20): ##Choose the layer that fit your application
    global intermediate_features
    intermediate_features = []
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)
    print(hook)
    with torch.no_grad():
        model(img)
    hook.remove()
    return intermediate_features[0]  # Access the first element of the list

# Make sure to preprocess the image since the input image must be 640x640x3
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    
    return img

# Load YOLOv8 model
weights_path = Path("path/to/your/model/best.pt")
model = YOLO(weights_path)
img = Path(r'##') #Drage your image path here

img = preprocess_image(img)
features = extract_features(model, img, layer_index=20)