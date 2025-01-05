import torch
import torch.nn as nn
import torch.optim.adam as Adam 
import torchvision
import tqdm
import numpy as np


Dataset_dir : str = r"/home/amzad/Desktop/Vqvae/dataset/"

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
    def __init__(self, config ,model, optimizer, train_loader):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion =  {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])
        
        self.train_loader = train_loader
        self.sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def trainner(self):
        recon_losses = []
        codebook_losses = []
        commitment_losses = []
        losses = []
 
        for epoch in range(self.config['epochs']):
            count = 0
            # pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for im, _ in tqdm(train_loader):
                im = im.float().to(self.device)
                self.optimizer.zero_grad()
                model_output = self.model(im)
                output = model_output['generated_image']
                quantize_losses = model_output['quantized_losses']

                if self.config['train_params']['save_training_image']:
                    torchvision.utils.save_image(
                        output,
                        f"../training_images/{count}.png",
                        normalize=True,
                        nrow=8,
                        range=(-1, 1),
                    )
                    count += 1
                   
                recon_loss = self.crtierion(output, im)
                loss = (self.config['train_params']['reconstruction_loss_weight']*recon_loss +
                self.config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'] +
                self.config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'])

                loss.backward()
                self.optimizer.step()
                recon_losses.append(recon_loss.item())
                codebook_losses.append(quantize_losses['codebook_loss'].item())
                commitment_losses.append(quantize_losses['commitment_loss'].item())
                losses.append(loss.item())
                print(
                    f"Epoch: {epoch+1}, Recon Loss: {recon_loss.item()/len(recon_loss)}, \
                    Codebook Loss: {quantize_losses['codebook_loss'].item()/len(quantize_losses['codebook_loss'])},\
                    Commitment Loss: {quantize_losses['commitment_loss'].item()/len(quantize_losses['commitment_loss'])},\
                    Loss: {loss.item()/len(loss)}" )
                mean_loss = np.mean(losses) 

                self.sheduler.step(mean_loss)

                inft = np.inf
                if mean_loss < inft:
                    inft = mean_loss
                    torch.save(self.model.state_dict(), self.config['model_params']['model_path'])
                    print("Model saved")



if __name__ == "__main__":
    from models import VQvae 
    from utils import read_config

    config_dir = r"/home/amzad/Desktop/Vqvae/config/config_v1.yaml"
    config = read_config(config_dir)
    

    # config = {
    #     'in_channels': [3, 16, 32, 8, 8] ,
    #     'latent_dim': 64,
    #     'transposebn_channels': [64, 32, 16, 3],
    #     'num_embeddings': 512,
    #     'embedding_dim': 64,
    #     'epochs': 100,
    #     'train_params': {
    #         'reconstruction_loss_weight': 1,
    #         'codebook_loss_weight': 0.25,
    #         'commitment_loss_weight': 0.25,
    #         'save_training_image': True
    #     },
    #     'model_params': {
    #         'model_path': 'model.pth'
    #     }
    # }

    model = VQvae(config)

    trainer = Trainer(config, model,  train_loader=train_loader)
    trainer.trainner()