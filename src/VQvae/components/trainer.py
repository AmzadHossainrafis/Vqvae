import torch
import torch.nn as nn
import torch.optim.adam as Adam 
import torchvision

Dataset_dir : str = r"/home/amzad/Desktop/Vqvae/dataset/arabic_mnist/"

transform = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#load the dataset 
train_dataset = torchvision.datasets.ImageFolder(
    root=Dataset_dir,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True
)


class Trainer:
    def __init__(self, config ,model, optimizer, criterion, train_loader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def trainner(self):
        pass 

