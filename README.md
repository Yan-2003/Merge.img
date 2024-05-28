
# Merge.img

Merge.img is a Python application that uses Tkinter for the GUI and PyTorch for deep learning tasks. The application performs neural style transfer, allowing you to merge the style of one image with the content of another image. This README provides an overview of the application's functionalities, requirements, how to run the code, and an explanation of how the code works.

## Functionalities
#### 1. GUI Creation:

- Utilizes Tkinter to create a graphical user interface.
- Allows users to select two images: one for style and one for content.
- Users can set the dimensions (height and width) for the images.
- Users can set the number of steps for the style transfer process.

#### 2. Image Processing:

- Uses PIL (Pillow) for image loading, conversion, and resizing.
- Transforms images into tensors suitable for PyTorch processing.
#### 3. Neural Style Transfer:

- Implements neural style transfer using the VGG19 model pre-trained on ImageNet.
- Defines loss functions for content and style.
- Optimizes the input image to combine the content of one image with the style of another.
#### 4. Saving and Displaying Results:

- Allows users to save the resultant image.
- Displays the progress of the rendering process within the application.
- Saves the output image in a specified directory.

## Requirements
#### Python Libraries
- `tkinter`: For creating the graphical user interface.
- `Pillow`: For image handling and processing.
- `torch`: For deep learning operations.
- `torchvision`: For model loading and transformations.
- `matplotlib`: For plotting images (if needed for display).

## Hardware
GPU is recommended for faster processing, but the code will fall back to CPU if GPU is not available.

# How to Run the Code

### Prerequisites
1. Install Python: Ensure Python 3.x is installed on your machine.
2. Install Required Libraries

```bash
pip install tkinter pillow torch torchvision matplotlib
````
### Execution

1. Save the Script: Save the provided script as `main.py`.
2. Run the Script: Execute the script using Python.
3. Using the Application:
- Select Images: Click on "Select Overlay Image" and "Select Input Image" to load the style and content images respectively.
- Set Parameters: Adjust the height, width, and steps as desired.
- Render: Click the "Render" button to start the style transfer process. The application will update the status and show the resultant image upon completion.
- Save Image: Use the "Save" button to save a copy of the selected image.

## Code Explanation
### Import and Setup
```python
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import time
import os
```
- **Tkinter**: For the GUI.
- **Pillow**: For image processing.
- **PyTorch and Torchvision**: For neural network operations and transformations.
- **Matplotlib**: For potential image plotting.
- **OS and Time**: For file operations and timing.

### Device Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
```
- Sets the computation device to GPU if available, otherwise defaults to CPU.
### Global Variables

```python
height = 300
width = 300
```

- Default image dimensions.

### Functions
`save_image()`
- Saves the selected image to the current directory.
`overlay_select_image()` and `input_select_image()`
- Opens file dialogs to select images and displays them in the application.
`render_image()`
- Main function for performing style transfer. It validates inputs, processes images, and uses PyTorch to perform neural style transfer.

### Style Transfer Implementation

#### Loss Functions
```python
class ContentLoss(nn.Module):
    # Calculates the content loss between the target and input images

class StyleLoss(nn.Module):
    # Calculates the style loss using Gram matrices
```

#### Model and Normalization
```python
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    # Normalizes the image based on mean and standard deviation
```

#### Style Transfer Process
```python
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    # Builds the model and calculates style and content losses

def get_input_optimizer(input_img):
    # Returns an optimizer for the input image

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img):
    # Runs the style transfer process and optimizes the input image

```

#### GUI Setup
```python
root = tk.Tk()
root.title("Merge.img")
# Set window size and properties

style = ttk.Style()
# Configure styles

frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Add buttons, labels, and entries for user interaction

root.mainloop()
# Starts the Tkinter event loop
```

### Developed a by:
- Julliane Tampus
- Cesar Ecleo
- Zeke Achas
#### As a final requirement for IT126-EP2
