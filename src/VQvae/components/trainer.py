import torch
import torch.nn as nn
import torch.optim.adam as Adam
import torchvision
import tqdm
import numpy as np


Dataset_dir: str = r"/home/amzad/Desktop/Vqvae/dataset/"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((31, 31)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# load the dataset
train_dataset = torchvision.datasets.ImageFolder(root=Dataset_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, num_workers=4, shuffle=True
)


class Trainer:
    def __init__(self, config, model, train_loader):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = {"l1": torch.nn.L1Loss(), "l2": torch.nn.MSELoss()}.get(
            config["crit"]
        )

        self.train_loader = train_loader
        self.sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, verbose=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def trainner(self):
        commitment_loss = []
        cookbook_loss = []
        recon = []

        for epoch in range(self.config["epochs"]):
            self.model.train()
            for i, (images, _) in enumerate(tqdm.tqdm(self.train_loader)):
                images = images.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
            
                recon_loss = self.criterion(output["decoder_output"], images)
                loss = (
                    self.config["recon_weight"] * recon_loss
                    + self.config["cookbook_wight"] * output["cookbook"]
                    + self.config["commitment_weight"]*output["comitment"]
                )

                commitment_loss.append(output["comitment"].item())
                cookbook_loss.append(output["cookbook"].item())
                recon.append(recon_loss.item())
                
                #



                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch: {epoch}, Commitment Loss: {np.mean(commitment_loss)}, Cookbook Loss: {np.mean(cookbook_loss)},recon_loss {recon_loss}"
            )
            if epoch % 2 == 0:
                # save the prediction image
                torchvision.utils.make_grid(output["decoder_output"]).permute(1, 2, 0)
                torchvision.utils.save_image(
                    output["decoder_output"],
                    f"output_epoch_{epoch}.png",
                    normalize=True,
                    
                )

            best_loss = np.inf
            mean_recon_loss = recon_loss.item()
            if mean_recon_loss < best_loss:
                best_loss = mean_recon_loss
                print(f"Model improved, saving model at {best_loss} loss, epoch {epoch}") 
                torch.save(self.model.state_dict(), "VQvae_best_model.pth")
            else:
                print("Model did not improve")

            self.sheduler.step(mean_recon_loss)


if __name__ == "__main__":

    from models import VQvae

    conf = {
        "in_channels": [3, 16, 32, 8, 8],
        "kernel_size": [3, 3, 3, 2],
        "kernel_strides": [2, 2, 1, 1],
        "convbn_blocks": 4,
        "latent_dim": 8,
        "transposebn_channels": [8, 8, 32, 16, 3],
        "transpose_kernel_size": [2, 3, 3, 3],
        "transpose_kernel_strides": [1, 1, 2, 2],
        "transpose_bn_blocks": 4,
        "num_embeddings": 512,
        "embedding_dim": 8,
        "commitment_cost": 0.25,
        "recon_weight": 5,
        "cookbook_wight": 1,
        "commitment_weight": 0.2,
        "crit": "l1",
        "epochs": 100,
    }

    model = VQvae(config=conf).to("cuda")
    trainer = Trainer(config=conf, model=model, train_loader=train_loader)
    trainer.trainner()
