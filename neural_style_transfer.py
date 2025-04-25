import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import copy

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader with fixed size
def load_image(image_path, size=512):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = GramMatrix()(target_feature).detach()
        self.gram = GramMatrix()
    def forward(self, input):
        G = self.gram(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Model with losses
def get_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# Run transfer
def run_style_transfer(cnn, content_img, style_img, input_img, steps=300, style_weight=1e6, content_weight=1e0):
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)
    for step in range(steps):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()
        input_img.data.clamp_(0, 1)
        if step % 50 == 0:
            print(f"Step {step}, Style Loss: {style_score.item():.2f}, Content Loss: {content_score.item():.2f}")
    return input_img

# Save output
def save_image(tensor, path):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

# ======== MAIN ========
# Replace with your own image paths
content_path = "content.jpeg"
style_path = "style.jpeg"
output_path = "output.jpg"

# Load images
content_img = load_image(content_path)
style_img = load_image(style_path)
input_img = content_img.clone()

# Load pre-trained VGG19
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Run style transfer
output = run_style_transfer(cnn, content_img, style_img, input_img)

# Save result
save_image(output, output_path)
print(f"Style transfer completed. Output saved as {output_path}")
