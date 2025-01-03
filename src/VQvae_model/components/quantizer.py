import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary 

class Quantizer(nn.Module):
    def __init__(self, config) -> None:
        super(Quantizer, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
        )
        self.embedding.weight.data.uniform_(
            -1 / config["num_embeddings"], 1 / config["num_embeddings"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        beta = 0.2
        dis = torch.cdist(x, self.embedding.weight.repeat(B, 1, 1))
        z = torch.argmin(dis, dim=-1)
        min_index = torch.index_select(self.embedding.weight, 0, z.view(-1)).view(
            B, H, W, C
        )


        comitment_loss = F.mse_loss(quantized.detach(), x)
        codebooke_loss = F.mse_loss(quantized, x.detach())
    
        quantizer_loss = beta * comitment_loss + codebooke_loss

        quantized = x + (min_index - x).detach()
        quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)
        min_index = min_index.reshape(B, C, H, W)

        return (
            quantized,
            {"cookbook_loss": codebooke_loss, "comitment_loss": comitment_loss},
            quantizer_loss,
            min_index,
        )


