from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from model import AutoEncoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image',type=str,help='Test Image')
args = parser.parse_args()

model = AutoEncoder(n_channels=1)

if torch.cuda.is_available():
	state_dict = torch.load('best_model.pt',weights_only=True)
else:
	state_dict = torch.load('best_model.pt',map_location=torch.device('cpu'),weights_only=True)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

with torch.no_grad():
    
    original_image = Image.open(args.image)
    grayscale_image = transforms.Grayscale(num_output_channels=1)(original_image)
    grayscale_image = transform(grayscale_image).unsqueeze(0)
    colored_recon = model(grayscale_image)
    plt.figure(figsize=(10,8))
    plt.subplot(131)
    plt.title('Grayscale image',fontweight='bold')
    plt.imshow(grayscale_image.squeeze(0).permute(1,2,0),cmap='gray')
    plt.subplot(132)
    plt.title('Original color image',fontweight='bold')
    plt.imshow(original_image.resize((128,128)))
    plt.subplot(133)
    plt.title('Reconstructed color image',fontweight='bold')
    plt.imshow(colored_recon.squeeze(0).permute(1,2,0))
    plt.tight_layout()
    plt.savefig(f'result_{args.image}.jpg',dpi=200)