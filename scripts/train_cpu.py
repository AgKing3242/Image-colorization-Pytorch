import torch
import torch.nn as nn
import torch.optim
import argparse
from model import AutoEncoder
from tqdm import tqdm
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,help='Training data directory')
parser.add_argument('--batch_size',type=int,help='Training batch size',default=32)
parser.add_argument('--epochs',type=int,help='Training epochs',default=10)
parser.add_argument('--lr',type=float,help='Initial learning rate',default=0.001)
parser.add_argument('--weight_decay',type=float,help='Weight decay for regularisation',default=1e-5)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataloader = DataLoader(CustomDataset(root_dir=args.data,transform=transform),batch_size=args.batch_size,shuffle=True)

epochs = args.epochs
learning_rate = args.lr
weight_decay = args.weight_decay
model = AutoEncoder(n_channels=1)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_loss = float('inf')
for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0

    for grayscale_image, color_image in tqdm(dataloader):

        prediction = model(grayscale_image)
        loss = criterion(prediction, color_image)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pt')

    print(f'Epoch: {epoch} Loss: {avg_loss:.6f} LR: {optimizer.param_groups[0]["lr"]:.6f}')

print('Model saved successfully.')