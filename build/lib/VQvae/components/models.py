import torch
import torch.nn as nn
from quantizer import Quantizer

class Encoder(nn.Module):
    '''
    Args:
        config (dict): Configuration dictionary containing:
            - in_channels (list): List of input channels for each convolutional layer.
            - kernel_size (list): List of kernel sizes for each convolutional layer.
            - kernel_strides (list): List of strides for each convolutional layer.
            - latent_dim (int): Dimension of the latent space.

    Attributes:
        encoder_block (nn.ModuleList): List of convolutional layers forming the encoder.
        latent_dim (int): Dimension of the latent space.
    '''
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
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=config["in_channels"][i + 1]),
                    nn.LeakyReLU(),
                )
                for i in range(config["convbn_blocks"] - 1)
            ]
        )

        self.encoder_block.append(
            nn.Conv2d(
                in_channels=config["in_channels"][-1],
                out_channels=config["in_channels"][-1],
                kernel_size=config["kernel_size"][-2],
                stride=config["kernel_strides"][-2],
                padding=1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Encoded tensor.
        """
        out = x
        for block in self.encoder_block:
            out = block(out)
        return out


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
                        stride=config["transpose_kernel_size"][i],
                        padding=0,
                    ),
                    nn.BatchNorm2d(num_features=config["transposebn_channels"][i + 1]),
                    nn.LeakyReLU(),
                )
                for i in range(config["transpose_bn_blocks"] - 1)
            ]
        )

        self.decoder_block.append(
            nn.Sequential(
                nn.Conv2d(
                    config["transposebn_channels"][-2],
                    config["transposebn_channels"][-1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.Tanh(),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Decoded tensor.
        """
        out = x
        for block in self.decoder_block:
            out = block(out)
        return out




class VQvae(nn.Module):
    def __init__(self, config) -> None:
        super(VQvae, self).__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.quantizer = Quantizer(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        quant_output, quant_loss, quant_idxs , min = self.quantizer(x)
        out = self.decoder(quant_output)
        # out = self.post_quantization_conv(x)
        return {
            "generated_image": out,
            "quantized_output": quant_loss['comitment_loss'],
            "quantized_losses": quant_loss['cookbook_loss'],
            "quantized_indices": quant_idxs,
            "min_index": min
        }


