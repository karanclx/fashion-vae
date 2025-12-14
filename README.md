# Variational Autoencoder (VAE) - Fashion MNIST

A clean, mathematically grounded implementation of a Variational Autoencoder (VAE) and $\beta$-VAE in PyTorch, trained on the Fashion-MNIST dataset. This repository focuses on a technically precise implementation of the Evidence Lower Bound (ELBO) loss and the reparameterization trick.

##  Model Architecture

![VAE Architecture](assets/vae_arch.png)


The model is composed of two probabilistic networks designed to map high-dimensional data to a continuous latent manifold.

* **Encoder :** Maps $28 \times 28$ grayscale inputs to a latent distribution. Instead of a single vector, it predicts a **mean ($\mu$)** and **log-variance ($\log\sigma^2$)** for each latent dimension.
* **Latent Space:** Modeled as a multivariate Gaussian distribution.
* **Reparameterization Trick:** Enables backpropagation through the stochastic sampling layer. We compute $z = \mu + \sigma \odot \epsilon$, where $\epsilon$ is fixed noise sampled from $\mathcal{N}(0, I)$.
* **Decoder :** Reconstructs the image from the latent sample $z$. It uses a **Sigmoid** activation function to output pixel probabilities, treating the normalized pixel values as Bernoulli distributions.

## Training Process & Objective

The model is trained to maximize the **Evidence Lower Bound (ELBO)** by minimizing the **Negative ELBO**.

$$\mathcal{L} = \text{BCE}(x, \hat{x}) + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

### Key Technical Details:
* **Reconstruction Loss:** Calculated as **Binary Cross Entropy (BCE)** between the input and reconstruction. Crucially, this is **summed** over all 784 pixels (not averaged) to maintain a gradient magnitude comparable to the KL term.
* **Regularization (KL Divergence):** Computed **analytically** per sample using the closed-form solution for Gaussians:
    $$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d_z} (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$
    This forces the latent distribution to approximate a standard Unit Gaussian $\mathcal{N}(0, I)$.
* **$\beta$-VAE Support:** The hyperparameter $\beta$ controls the trade-off:
    * $\beta = 1.0$: Standard VAE formulation.
    * $\beta > 1.0$: Enforces stronger disentanglement of latent factors (at the cost of some reconstruction quality).
* **Optimization:** Trained using the **Adam** optimizer (lr=1e-3). Convergence on Fashion-MNIST typically occurs within 50–100 epochs as the ELBO plateaus.

##  Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/vae-fashion-mnist.git](https://github.com/yourusername/vae-fashion-mnist.git)
    cd vae-fashion-mnist
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

To train the model from scratch:

```bash
python train.py --epochs 50 --batch_size 64 --beta 1.0
