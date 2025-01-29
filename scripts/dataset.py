from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class CustomDataset(Dataset):

    def __init__(self,root_dir,transform=None):

        self.transform = transform
        self.img_paths = []
        for path in os.listdir(root_dir):
            self.img_paths.append(os.path.join(root_dir,path))

    def __len__(self):

        return len(self.img_paths)

    def __getitem__(self,index):

        color_image = Image.open(self.img_paths[index]).convert('RGB')
        grayscale_image = transforms.Grayscale(num_output_channels=1)(color_image)

        if self.transform:

            color_image = self.transform(color_image)
            grayscale_image = self.transform(grayscale_image)

        return grayscale_image,color_image