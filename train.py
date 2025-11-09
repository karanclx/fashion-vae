from pyimagesearch import config, network, utils
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import os
from torch.optim import lr_scheduler

import matplotlib
device = torch.device(config.DEVICE)   
matplotlib.use("agg") # change the backend based on the non gui backend available

# define the transformation to be applied to the data
transform = transforms.Compose(
    [transforms.Pad(padding = 2), transforms.ToTensor()],
)

# load the fashionmnist training data and create a dataloader
trainset = datasets.FashionMNIST(
    "data", train = True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.BATCH_SIZE, shuffle=True
)

# load the Fashionmnist test set and create a dataloader
testset = datasets.FashionMNIST(
    "data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=False
)
model = network.VAE(
    input_channels=config.CHANNELS,
    hidden_dim=400,                    # ← add HIDDEN_DIM to config.py if needed
    latent_dim=config.EMBEDDING_DIM
).to(device)


# instantiate optimiser and scheduler
optimizer = optim.Adam(
    model.parameters(), lr=config.LR
)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer ,mode="min", factor=0.1, patience=config.PATIENCE
)

# initiaize the best validation loss as infinity 
best_val_loss = float("inf")

# start training by looping over the number of epochs
for epoch in range(1, config.EPOCHS + 1):
    #set the model model to train mode
    #and move it to CPU/GPU

    model.train()
    

    train_loss = 0.0
    
    # loop over batches of the training dataset
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # forward pass through the model
        mu, logvar, recon = model(data)  # mu first, then logvar, then recon 

        if batch_idx == 0: 
            print(f"DEBUG - First batch:")
            print(f"  Input data shape: {data.shape}")
            print(f"  mu shape: {mu.shape}")
            print(f"  logvar shape: {logvar.shape}")
            print(f"  recon shape: {recon.shape}")
            print(f"  Model type: {type(model)}")
            print(f"  Encoder output check...")
            test_z_mean, test_z_logvar, test_z = model.encoder(data)
            print(f"    z_mean: {test_z_mean.shape}, z: {test_z.shape}")
            test_recon = model.decoder(test_z)
            print(f"    decoder output: {test_recon.shape}")
            
        

        #compute the model loss
        
        loss = utils.vae_loss(recon, data, mu, logvar)

        # backwaed pass and optimizer step
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)  # Average per batch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            mu, logvar, recon = model(data)
            
            loss = utils.vae_loss(recon, data, mu, logvar)
            val_loss += loss.item()

    # average val loss over all batches
    val_loss /= len(test_loader)

    scheduler.step(val_loss)

    #print every 20 epochs or at the end
    if epoch % 20 == 0 or epoch + 1 == config.EPOCHS:
        print(f"Epoch {epoch:03d} | Train : {train_loss:.4f} | Val : {val_loss:.4f}")     

    # saving the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "vae_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "optimizer_state_dict": optimizer.state_dict()

        }, config.MODEL_WEIGHTS_PATH)
        print(f"  -> New best model saved! (val_loss: {val_loss:.4f})")

print(f"training complete, best model saved at:{config.MODEL_WEIGHTS_PATH}")                