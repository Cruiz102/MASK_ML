import torch
import torch.nn as nn

class ImageAutoEncoder(nn.Module):
    def __init__(self,
                 image_encoder: nn.Module,
                 decoder: nn.Module,
                 image_size: int,
                 flatten: bool

                 ):
        super(ImageAutoEncoder, self).__init__()
        self.image_encoder = image_encoder
        self.decoder = decoder
        self.image_size = image_size
        self.flatten = flatten 
    def forward(self, x):
        batch_size = x.shape[0]
        original_shape = x.shape  # Store the original shape
        
        if self.flatten:
            x = x.view(batch_size, -1)
        
        # Encode and decode
        latent = self.image_encoder(x)
        reconstructed = self.decoder(latent)
        
        # Reshape back to original dimensions if flattened
        if self.flatten:
            reconstructed = reconstructed.view(original_shape)
        
        return reconstructed
    

class MaskPatchEncoder(nn.Module):
    def __init__(self, mask_ratio: float = 0.75,
                 mask_strag: str = "random",
                 image_size: int = 256,
                 patch_size: int = 16,
                 num_channels: int = 3,
                 embedded_size: int = 200,
                 interpolation: bool = False,
                 interpolation_scale: int = 1
                 ):
        super(MaskPatchEncoder, self).__init__()
        # Save parameters
        self.mask_ratio = mask_ratio
        self.mask_strag = mask_strag
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedded_size = embedded_size
        self.interpolation = interpolation
        self.interpolation_scale = interpolation_scale
        
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embedded_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedded_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Randomly mask patches
        num_masked = int(self.num_patches * self.mask_ratio)
        indices = torch.randperm(self.num_patches)
        masked_indices = indices[:num_masked]
        unmasked_indices = indices[num_masked:]

        # Mask out the patches
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        mask[masked_indices] = True

        return x[:, ~mask, :], masked_indices, unmasked_indices


class MaskedAutoEncoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 mask_ratio: float = 0.75,
                 image_size: int = 256,
                 patch_size: int = 16,
                 num_channels: int = 3,
                embedded_size: int = 200
                 ):
        super(MaskedAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_token = nn.Parameter(torch.zeros(1, encoder.embedded_size))
        self.patch_encoder = MaskPatchEncoder(mask_ratio=mask_ratio, 
                                              image_size=image_size, 
                                              patch_size=patch_size, 
                                              num_channels=num_channels, 
                                              embedded_size=embedded_size)

    def forward(self, x):
        patch_embeddings, masked_indices, unmasked_indices = self.patch_encoder(x)

        encoded_visible = self.encoder(patch_embeddings)

        batch_size, num_patches, d_model = x.shape[0], self.patch_encoder.num_patches, self.patch_encoder.embedded_size
        full_decoder_input = torch.zeros((batch_size, num_patches, d_model), device=x.device)
        full_decoder_input[:, unmasked_indices, :] = encoded_visible
        full_decoder_input[:, masked_indices, :] = self.mask_token.unsqueeze(0).repeat(batch_size, len(masked_indices), 1)

        full_decoder_input += self.patch_encoder.pos_embed
        reconstructed_patches = self.decoder(full_decoder_input)

        reconstructed_image = torch.zeros_like(x)
        patch_size = self.patch_encoder.patch_size
        idx = 0
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                reconstructed_image[:, :, i:i+patch_size, j:j+patch_size] = reconstructed_patches[:, idx, :].view(-1, x.size(1), patch_size, patch_size)
                idx += 1

        return reconstructed_image, masked_indices


class SparseAutoencoderCriterion(nn.Module):
    def __init__(self, beta=1.0, sparsity_target=0.05):
        super(SparseAutoencoderCriterion, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.beta = beta  # Weight for sparsity penalty
        self.sparsity_target = sparsity_target  # Desired average activation

    def forward(self, outputs, inputs, activations):
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(outputs, inputs)

        # Compute sparsity loss
        mean_activation = torch.mean(activations, dim=0)  # Mean activation for each latent unit
        sparsity_loss = torch.sum(self.sparsity_target * torch.log(self.sparsity_target / mean_activation) +
                                  (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - mean_activation)))

        # Total loss
        loss = recon_loss + self.beta * sparsity_loss
        return loss
    

class DenoisingAutoencoderCriterion(nn.Module):
    def __init__(self, noise_std=0.1):
        """
        Initialize the denoising autoencoder criterion.
        :param noise_std: Standard deviation of the Gaussian noise to add.
        """
        super(DenoisingAutoencoderCriterion, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.noise_std = noise_std

    def forward(self, model, clean_inputs):
        """
        Forward method computes the loss by adding noise and comparing with clean inputs.
        :param model: The autoencoder model.
        :param clean_inputs: Clean input images.
        :return: Denoising loss.
        """
        # Add noise to the inputs
        noisy_inputs = clean_inputs + torch.randn_like(clean_inputs) * self.noise_std
        noisy_inputs = torch.clamp(noisy_inputs, 0, 1)  # Clamp to valid image range [0, 1]

        # Forward pass through the model
        outputs = model(noisy_inputs)

        # Compute reconstruction loss
        loss = self.reconstruction_loss(outputs, clean_inputs)
        return loss



class ContractiveAutoencoderCriterion(nn.Module):
    def __init__(self, lambda_reg=1.0):
        """
        Initialize the Contractive Autoencoder Criterion.
        :param lambda_reg: Regularization weight for the contractive penalty.
        """
        super(ContractiveAutoencoderCriterion, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, model, inputs):
        """
        Forward method to compute the loss.
        :param model: The autoencoder model.
        :param inputs: Original inputs.
        :return: Contractive loss.
        """
        inputs.requires_grad = True
        
        latent = model.image_encoder(inputs)
        reconstruction = model(inputs)
        recon_loss = self.reconstruction_loss(reconstruction, inputs)

        # Compute the Jacobian of the latent representation
        # Sum the latent representations to aggregate gradients
        latent_sum = torch.sum(latent)
        latent_sum.backward(retain_graph=True)

        # Compute the Frobenius norm of the Jacobian for the contractive loss
        contractive_loss = 0
        for param in inputs.grad:
            contractive_loss += torch.sum(param ** 2)
        
        loss = recon_loss + self.lambda_reg * contractive_loss
        inputs.requires_grad = False

        return loss


def collate_fn(batch):
    """
    Custom collate function to handle varying sizes in COCO dataset.
    Resizes images and keeps annotations as-is.
    """
    images = []
    targets = []
    for image, target in batch:
        # Images are already tensors because of the dataset transform
        images.append(image)  
        targets.append(target)  # Keep annotations as they are
    return torch.stack(images), targets

