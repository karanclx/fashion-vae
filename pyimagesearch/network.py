import torch
import torch.nn as nn
import torch.nn.functional as F 
#torch.nn.functional provides functions (useful for operations which dont have any params eg activation functions, certain loss functions)
#torch.nn provides classes 
from torch.distributions.normal import Normal
# importing Normal Class from torch.distributions which provides functionalities to create a manipulate gaussian distributions.


#defining a class for Sampling , this class will be used in the encoder for sampling in the latent space 

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.shape # get the shape of the tensor for the mean and log var

        epsilon = Normal(0,1).sample((batch,dim)).to(z_mean.device)
        # generate a normal random tensor with the same shape as z_mean to be used in the reparameterisation trick



        return z_mean + torch.exp(0.5 * z_log_var) * epsilon # applying the reparameterisation trick 
    

# defining the encoder
class Encoder(nn.Module):
    def __init__(self, image_size, embedding_dim):
        super(Encoder, self).__init__()

        # define the convolutional layers for downsampling and feature extraction

        self.conv1 = nn.Conv2d(1,32,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,stride=2,padding=1)
        
        # defining a flatten layer to flatten the tensor before feeding it into the desired embedding dimensions
        self.flatten = nn.Flatten()
        # define fully connected layers to transform the tensor into the desired embedding dimensions

        self.fc_mean = nn.Linear(
            128 * (image_size // 8)*(image_size // 8), embedding_dim
        )
        self.fc_log_var = nn.Linear(
            128 * (image_size // 8) * (image_size // 8), embedding_dim)
        
        self.sampling = Sampling()

    def forward(self, x):
        # x here represents one bacth of images
        # apply convolutional layers with relu activation function
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten the tensor
        x = self.flatten(x)

        # get the mean and log variance of the latent space distribution
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        # sample a latent vector using the reparam trick
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z
       
# defining the decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()


        self.shape_before_flattening = shape_before_flattening

        channels, height, width = shape_before_flattening
        flattened_size = channels * height * width

        #define a fc layer to transform the latent vector back to the shape before flattening
        self.fc = nn.Linear(
            embedding_dim,
            flattened_size
        )
        # define a reshape function to reshape the tensor back to its original shape
        self.reshape = lambda x : x.view(-1, *shape_before_flattening)
        # define the transposed convolutionla layers for the decoder to upsample and generate the reconstructed image

        self.deconv1 = nn.ConvTranspose2d(
            128,64,3,stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64,32,3,stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            32,1,3,stride=2, padding=1, output_padding=1
        )
    def forward(self, x):
        #pass the latent vector through the fc layer
        x = F.relu(self.fc(x))
         # reshape the tensor
        x = self.reshape(x)

        # apply transposed convolutions layers for the tensor back to its original shape
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)   # the final transposed convolutional layer with a sigmoid activation to generate the final output 

        x = torch.sigmoid(x)
        return x
    


# defining the VAE class
class VAE(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()

        # Use latent_dim as embedding_dim
        encoder = Encoder(image_size=32, embedding_dim=latent_dim)
        decoder = Decoder(
            embedding_dim=latent_dim,
            shape_before_flattening=(128, 32 // 8, 32 // 8)  # (128, 4, 4)
        )
        self.encoder = encoder
        self.decoder = decoder



    def forward(self, x):
        # passing the input through the encoder to get the latent vector
        z_mean, z_log_var, z = self.encoder(x) 
        # passing the latent vector through the decoder to get the reconstructed image
        reconstruction = self.decoder(z)

        # return the mean and log_var of the reconstructed image
        return z_mean, z_log_var, reconstruction  


      
           


        
        
