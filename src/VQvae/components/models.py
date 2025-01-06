import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder class for Vector Quantized Variational Autoencoder (VQ-VAE).

    Args:
        config (dict): Configuration dictionary containing:
            - in_channels (list): List of input channels for each convolutional layer.
            - kernel_size (list): List of kernel sizes for each convolutional layer.
            - kernel_strides (list): List of strides for each convolutional layer.
            - latent_dim (int): Dimension of the latent space.

    Attributes:
        encoder_block (nn.ModuleList): List of convolutional layers forming the encoder.
        latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, config) -> None:
        super(Encoder, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]

        self.encoder_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=config["in_channels"][i],
                        out_channels=config["in_channels"][i + 1],
                        kernel_size=config["kernel_size"][i],
                        stride=config["kernel_strides"][i],
                    ),
                    nn.BatchNorm2d(config["in_channels"][i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(config["kernel_size"]))
            ]
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Encoded tensor.
        """
        for layer in self.encoder_block:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """
    Decoder class for Vector Quantized Variational Autoencoder (VQ-VAE).

    Args:
        config (dict): Configuration dictionary containing:
            - transposebn_channels (list): List of input channels for each transposed convolutional layer.
            - transpose_kernel_size (list): List of kernel sizes for each transposed convolutional layer.
            - transpose_kernel_strides (list): List of strides for each transposed convolutional layer.
            - latent_dim (int): Dimension of the latent space.

    Attributes:
        decoder_block (nn.ModuleList): List of transposed convolutional layers forming the decoder.
        latent_dim (int): Dimension of the latent space.
    """

    def __init__(self, config) -> None:
        super(Decoder, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]

        self.decoder_block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=config["transposebn_channels"][i],
                        out_channels=config["transposebn_channels"][i + 1],
                        kernel_size=config["transpose_kernel_size"][i],
                        stride=config["transpose_kernel_strides"][i],
                    ),
                    nn.BatchNorm2d(config["transposebn_channels"][i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(config["transpose_kernel_size"]))
            ]
        )

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Decoded tensor.
        """
        for layer in self.decoder_block:
            x = layer(x)
        return x


from quantizer import Quantizer


class VQvae(nn.Module):
    def __init__(self, config) -> None:
        super(VQvae, self).__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.quantizer = Quantizer(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        quant_output, c, v, b = self.quantizer(x)
        out = self.decoder(x)

        return {
            "decoder_output": out,
            "cookbook": c["cookbook_loss"],
            "comitment": c["comitment_loss"],
            "quantizer_loss": v,
            "min_index": b,
        }

    # if __name__ =="__main__":

    # config = {
    #     'in_channels': [3, 16, 32, 8, 8] ,
    #     'kernel_size': [3,3,3,2],
    #     'kernel_strides': [2, 2, 1, 1],
    #     'convbn_blocks': 4,
    #     'latent_dim': 8,
    #     'transposebn_channels': [8, 8, 32, 16, 3],
    #     'transpose_kernel_size': [1,2,2,2],
    #     'transpose_kernel_strides': [1,2,1,1],
    #     'transpose_bn_blocks': 4
    # }

    # config = {
    #     "in_channels": [3, 16, 32, 8, 8],
    #     "kernel_size": [3, 3, 3, 2],
    #     "kernel_strides": [2, 2, 1, 1],
    #     "convbn_blocks": 4,
    #     "latent_dim": 8,
    #     "transposebn_channels": [8, 8, 32, 16, 3],
    #     "transpose_kernel_size": [2, 3, 3, 3],
    #     "transpose_kernel_strides": [1, 1, 2, 2],
    #     "transpose_bn_blocks": 4,
    #     "num_embeddings": 512,
    #     "embedding_dim": 8,  # do not change this
    # }
