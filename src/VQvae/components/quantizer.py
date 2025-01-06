import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """
    Quantizer class for Vector Quantized Variational Autoencoder (VQ-VAE).

    Args:
        config (dict): Configuration dictionary containing:
            - num_embeddings (int): Number of embeddings in the codebook.
            - embedding_dim (int): Dimension of each embedding vector.
            - commitment_cost (float): Weight for the commitment loss term.

    Attributes:
        embeddings (nn.Embedding): Embedding layer representing the codebook.
    """

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
        """
        Forward pass through the quantizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            quantized (torch.Tensor): Quantized tensor.
            loss (torch.Tensor): Quantization loss.
            encoding_indices (torch.Tensor): Indices of the closest embeddings.
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, -1, C)
        beta = 0.2
        dis = torch.cdist(x, self.embedding.weight[None:].repeat(B, 1, 1))

        z = torch.argmin(dis, dim=-1)

        min_index = torch.index_select(self.embedding.weight, 0, z.view(-1)).view(
            B, H, W, C
        )
        x = x.view(B, min_index.size(1), min_index.size(2), min_index.size(3))
        comitment_loss = torch.mean((x - min_index.detach()) ** 2)
        codebooke_loss = torch.mean((x.detach() - min_index) ** 2)
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


# remark : this config working fine with the model
# if __name__ == "__main__":
#     from models import Encoder, Decoder , VQvae
#     config = {
#         'in_channels': [3, 16, 32, 8, 8] ,
#         'kernel_size': [3,3,3,2],
#         'kernel_strides': [2, 2, 1, 1],
#         'convbn_blocks': 4,
#         'latent_dim': 8,
#         'transposebn_channels': [8, 8, 32, 16, 3],
#         'transpose_kernel_size': [2,3,3,3],
#         'transpose_kernel_strides': [1,1,2,2],
#         'transpose_bn_blocks': 4,
#         "num_embeddings": 512,
#         "embedding_dim": 8,
#         "commitment_cost": 0.25,
#     }

#     x = torch.randn(1, 3, 32, 32)
#     model = VQvae(config)
#     out = model(x)
#     print(out['decoder_output'].shape)
