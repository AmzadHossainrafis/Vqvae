import torch
import torch.nn as nn
from torchinfo import summary


class Encoder(nn.Module):
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
        out = x
        for block in self.encoder_block:
            out = block(out)
        return out


class Decoder(nn.Module):
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
        out = x
        for block in self.decoder_block:
            out = block(out)
        return out
from quantizer import Quantizer


class VQvae(nn.Module):
    def __init__(self, config) -> None:
        super(VQvae, self).__init__()
        self.config = config

        self.encoder = Encoder(config)
        pre_quantization_conv = nn.Conv2d(
            in_channels=config["in_channels"][-1],
            out_channels=config["latent_dim"],
            kernel_size=1,
            padding=1,
        )
        self.quantizer = Quantizer(config)
      
        self.post_quantization_conv = nn.Conv2d(
            in_channels=config["latent_dim"],
            out_channels=config["in_channels"][-1],
            kernel_size=1,
            padding=1,
        )

        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pre_quantization_conv(x)
        quant_output, quant_loss, quant_idxs = self.quantizer(x)
        x = self.post_quantization_conv(quant_output)
        x = self.decoder(x)
        out = self.post_quantization_conv(x)
        return {
            'generated_image' : out,
            'quantized_output' : quant_output,
            'quantized_losses' : quant_loss,
            'quantized_indices' : quant_idxs
        }

# if __name__ =="__main__":
#     config = {
#         'in_channels': [3, 16, 32, 8, 8] ,
#         'kernel_size': [3,3,3,2],
#         'kernel_strides': [2, 2, 1, 1],
#         'convbn_blocks': 4,
#         'latent_dim': 8,
#         'transposebn_channels': [8, 8, 32, 16, 3],
#         'transpose_kernel_size': [1,2,2,2],
#         'transpose_kernel_strides': [1,2,1,1],
#         'transpose_bn_blocks': 4
#     }

#     # config = {
#     #     'in_channels': [3, 16, 32, 8, 8] ,
#     #     'kernel_size': [3,3,3,2],
#     #     'kernel_strides': [2, 2, 1, 1],
#     #     'convbn_blocks': 4,
#     #     'latent_dim': 8,
#     #     'transposebn_channels': [8, 8, 32, 16, 3],
#     #     'transpose_kernel_size': [2,3,3,3],
#     #     'transpose_kernel_strides': [1,1,2,2],
#     #     'transpose_bn_blocks': 4
#     # }


#     encoder = Encoder(config).to('cuda')
#     decoder = Decoder(config).to('cuda')
#     summary(encoder, input_size=(1, 3, 128, 128))
#     summary(decoder, input_size=(1, 8, 32, 32))
