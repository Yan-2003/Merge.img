import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os


""" this is the import for the touch part """
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights 

import copy
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


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
        image = Image.open(overlay_selected_image_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        overlay_img_label.config(image=photo)
        overlay_img_label.image = photo

def input_select_image():
    global input_selected_image_path
    input_selected_image_path = filedialog.askopenfilename()
    if input_selected_image_path:
        image = Image.open(input_selected_image_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        input_img_label.config(image=photo)
        input_img_label.image = photo


def render_image():


    redner_label.config(text="Rendering...")


    imsize = 521 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(overlay_selected_image_path)
    content_img = image_loader(input_selected_image_path)

    #assert style_img.size() == content_img.size()

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

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1,1,1)
        
        def forward(self, img):
            return (img - self.mean) / self.std
        
    


    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std)

        # just in order to have an iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
    
    input_img = content_img.clone()
    # if you want to use white noise by using the following code:
    #
    # .. code-block:: python
    #
    #   input_img = torch.randn(content_img.data.size())

    # add the original input image to the figure:

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer
    
    def run_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=int(entry.get()),
                        style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        redner_label.config(text="Building the style transfer model..")
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        optimizer = get_input_optimizer(input_img)

        redner_label.config(text="Optimizing...")
        run = [0]
        while run[0] <= num_steps:
            root.update_idletasks()
            def closure():
                # correct the values of updated input image
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
                    redner_label.config(text="run {}:".format(run) + " " + 'Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print("run {}:".format(run) + " " + 'Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                return style_score + content_score
            root.update_idletasks()
            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    plt.figure()
    plt.ioff()
    root.update_idletasks()
    redner_label.config(text="Render Finish....")
    out_t = (output.data.squeeze())
    output_img = transforms.ToPILImage()(out_t)
    output_img.save('./render/' + str(int(time.time())) +'output.png')
    output_img






root = tk.Tk()
root.title("Merge.img")
icon_path = "icon.ico"
root.iconbitmap(icon_path)

window_width = 800
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width/2) - (window_width/2)
y_coordinate = (screen_height/2) - (window_height/2)
root.geometry("%dx%d+%d+%d" % (window_width, window_height, x_coordinate, y_coordinate))
root.resizable(False, False)




overlay_img_button = tk.Button(root, text="Select Overlay Image", command=overlay_select_image)
overlay_img_button.place(x=2, y=1)

input_img_button = tk.Button(root, text="Select Input Image", command=input_select_image)
input_img_button.place(x=125, y=1)

default_value = tk.StringVar(value="1000")



scale_input = tk.Label(root, text="Scale Plot (Note: the more the plot is greater the longer it render.)")
scale_input.place(x=5 , y=380)
entry = tk.Entry(root, textvariable=default_value)
entry.place(x=10, y=400)


render_button = tk.Button(root, text="Render Image", command=render_image)
render_button.place(x=10, y=430)


""" save_button = tk.Button(root, text="Save Image", command=save_image)
save_button.pack(pady=5) """


overlay_img_label = tk.Label(root)
input_img_label = tk.Label(root)

overlay_img_label.place(x=2,y=50)
input_img_label.place(x=305, y=50)
redner_label = tk.Label(root, text="")
redner_label.place(x=10, y=480)




root.mainloop()