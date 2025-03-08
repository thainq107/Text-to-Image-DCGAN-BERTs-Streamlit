import os
import torch
import gdown
import numpy as np
import torch.nn as nn
import streamlit as st
from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer("all-mpnet-base-v2")

class Generator(nn.Module):
    
    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator, self).__init__()
        self.reduced_dim_size = reduced_dim_size

        #768-->256
        self.textEncoder = nn.Sequential(
            nn.Linear(in_features = embedding_size, out_features = reduced_dim_size),
            nn.BatchNorm1d(num_features = reduced_dim_size),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

        self.upsamplingBlock = nn.Sequential(
            #256+100 --> 1024
            nn.ConvTranspose2d(noise_size + reduced_dim_size, feature_size * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            # 1024 --> 512
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),

            # 512 --> 256
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),

            # 256 --> 128
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # 128 --> 128
            nn.ConvTranspose2d(feature_size, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # 128 --> 3
            nn.ConvTranspose2d(feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, noise, text_embeddings):
        encoded_text = self.textEncoder(text_embeddings)
        concat_input = torch.cat([noise, encoded_text], dim = 1).unsqueeze(2).unsqueeze(2)
        output = self.upsamplingBlock(concat_input)
        return output

generator = Generator(100, 128, 3, 768, 256)

model_name = 'generator.pth'
if not os.path.exists(model_name):
    gdown.download(id="1EH7B-4w7cLY0RdoZHi6q_zR3ATU49TXL")
generator.load_state_dict(torch.load(model_name))
generator.eval()

def main():
    st.title('Text To Image Using DCGAN-BERTs')
    st.header('Model: DCGAN. Dataset: Flower. Text Encoder: BERTs')
    text_input = st.text_input("Sentence: ", "this pale pink flower has a large yellow and green pistil.")

    embed_caption = torch.tensor(bert_model.encode(text_input))
    noise = torch.randn(size=(1, 100))
    embed_caption = embed_caption.unsqueeze(0)
    with torch.no_grad():
        test_images = generator(noise, embed_caption)
    grid = torchvision.utils.make_grid(test_images, normalize=True)
    st.success(show_grid(grid))

if __name__ == '__main__':
     main() 
