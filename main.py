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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

height = 300
width = 300

def save_image():
    global selected_image_path
    if selected_image_path:
        save_path = './' + "_copy.jpg"
        image = Image.open(selected_image_path)
        image.save(save_path)
        print("Image saved as:", save_path)

def overlay_select_image():
    global overlay_selected_image_path
    overlay_selected_image_path = filedialog.askopenfilename()
    if overlay_selected_image_path:
        try:
            image = Image.open(overlay_selected_image_path).convert('RGB')
            image = image.resize((height, width), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            overlay_img_label.config(image=photo)
            overlay_img_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

def input_select_image():
    global input_selected_image_path
    input_selected_image_path = filedialog.askopenfilename()
    if input_selected_image_path:
        try:
            image = Image.open(input_selected_image_path).convert('RGB')
            image = image.resize((height, width), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            input_img_label.config(image=photo)
            input_img_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

def render_image():
    global height, width
    try:
        height = int(height_input.get())
        width = int(width_input.get())
    except ValueError:
        messagebox.showerror("Error", "Height and Width must be integers.")
        return

    redner_label.config(text="Rendering...")

    if not overlay_selected_image_path or not input_selected_image_path:
        messagebox.showwarning("Image Selection", "Please select both overlay and input images.")
        return

    imsize = 521 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name).convert('RGB')
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(overlay_selected_image_path)
    content_img = image_loader(input_selected_image_path)

    if style_img.size() != content_img.size():
        messagebox.showwarning("Image Size Mismatch", "The size of overlay and input image do not match.")
        return

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.view(-1, 1, 1)
            self.std = std.view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=int(steps_entry.get()),
                           style_weight=1000000, content_weight=1):
        redner_label.config(text="Building the style transfer model...")
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        input_img.requires_grad_(True)
        model.eval()
        model.requires_grad_(False)
        optimizer = get_input_optimizer(input_img)

        redner_label.config(text="Optimizing...")
        run = [0]
        while run[0] <= num_steps:
            root.update_idletasks()
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                    root.update_idletasks()
                for cl in content_losses:
                    root.update_idletasks()
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    root.update_idletasks()
                    redner_label.config(text=f"run {run[0]}: Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
                    print(f"run {run[0]}: Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
                return style_score + content_score

            root.update_idletasks()
            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    plt.figure()
    plt.ioff()
    root.update_idletasks()
    redner_label.config(text="Render Finish....")
    out_t = output.data.squeeze()
    output_img = transforms.ToPILImage()(out_t)
    os.makedirs('./render', exist_ok=True)
    output_img.save(f'./render/{int(time.time())}output.png')
    return messagebox.showinfo("Render Finish", "Your render is done.")

root = tk.Tk()
root.title("Merge.img")
icon_path = "./icon.ico"
root.iconbitmap(icon_path)

window_width = 800
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width/2) - (window_width/2)
y_coordinate = (screen_height/2) - (window_height/2)
root.geometry(f"{window_width}x{window_height}+{int(x_coordinate)}+{int(y_coordinate)}")
root.resizable(False, False)

style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#2E2E2E')
style.configure('TLabel', background='#2E2E2E', foreground='#FFFFFF')
style.configure('TButton', background='#454545', foreground='#FFFFFF', borderwidth=1)
style.map('TButton', background=[('active', '#666666')])

frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

overlay_img_button = ttk.Button(frame, text="Select Overlay Image", command=overlay_select_image)
overlay_img_button.grid(row=0, column=0, pady=5)

input_img_button = ttk.Button(frame, text="Select Input Image", command=input_select_image)
input_img_button.grid(row=0, column=1, pady=5, padx=5)

height_label = ttk.Label(frame, text="Height:")
height_label.grid(row=1, column=0, pady=5, sticky=tk.W)
height_input = ttk.Entry(frame, width=10)
height_input.grid(row=1, column=1, pady=5, sticky=tk.W)
height_input.insert(0, str(height))

width_label = ttk.Label(frame, text="Width:")
width_label.grid(row=2, column=0, pady=5, sticky=tk.W)
width_input = ttk.Entry(frame, width=10)
width_input.grid(row=2, column=1, pady=5, sticky=tk.W)
width_input.insert(0, str(width))

steps_label = ttk.Label(frame, text="Steps:")
steps_label.grid(row=3, column=0, pady=5, sticky=tk.W)
steps_entry = ttk.Entry(frame, width=10)
steps_entry.grid(row=3, column=1, pady=5, sticky=tk.W)
steps_entry.insert(0, "300")

save_button = ttk.Button(frame, text="Save", command=save_image)
save_button.grid(row=4, column=0, pady=5, sticky=tk.W)

render_button = ttk.Button(frame, text="Render", command=render_image)
render_button.grid(row=4, column=1, pady=5, sticky=tk.W)

overlay_img_label = ttk.Label(frame)
overlay_img_label.grid(row=5, column=0, pady=10, padx=5, columnspan=2)

input_img_label = ttk.Label(frame)
input_img_label.grid(row=5, column=1, pady=10, padx=5, columnspan=2)

redner_label = ttk.Label(frame, text="Render Info....")
redner_label.grid(row=6, column=0, pady=10, padx=5, columnspan=2)

root.mainloop()
