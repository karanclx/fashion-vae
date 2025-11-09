import os
import torch

DEVICE = (
 "mps" if torch.backends.mps.is_available() else "cpu"
)

LR = 0.001
PATIENCE = 2
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 64
EMBEDDING_DIM = 2
EPOCHS = 100
SHAPE_BEFORE_FLATTENING = (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)


# creating output directory

output_dir = "output"
os.makedirs("output", exist_ok=True)

training_progress_dir = os.path.join(output_dir, "training_process")
os.makedirs(training_progress_dir, exist_ok=True)

#create the model_weights directory inside the output directory for storing the weights of our vae
model_weights_dir = os.path.join(output_dir, "model_weights")
#define model_weights, reconstruction & real before training images paths

MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae.pt")

# before training images
FILE_RECON_BEFORE_TRAINING = os.path.join(
    output_dir, "reconstruct_before_train.png"
)
FILE_REAL_BEFORE_TRAINING = os.path.join(
    output_dir, "real_test_images_before_train.png"
)

# define reconstruction  & real after training images paths
#after training images
FILE_RECON_AFTER_TRAINING = os.path.join(
    output_dir, "reconstruct_after_train.png"
)
FILE_REAL_AFTER_TRAINING = os.path.join(
    output_dir, "real_test_images_after_train.png"
)

# define latent space and image grid embedding plot paths
LATENT_SPACE_PLOT = os.path.join(output_dir, "embedding_visualize.png")
IMAGE_GRID_EMBEDDINGS_PLOT = os.path.join(
    output_dir, "image_grid_on_embeddings.png"
)

# define linearly and normally sampled latent space reconstructions plot paths

LINEARLY_SAMPLED_RECONSTRUCTIONS_PLOT = os.path.join(
    output_dir, "linearly_sampled_reconstructions.png"

)
NORMALLY_SAMPLED_RECONSTRUCTIONS = os.path.join(
    output_dir, "normally_sampled_reconstructions.png"
)

# defining class labels dictionary

CLASS_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",

}