import torch
import torch.nn as nn

class ImageAutoEncoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 image_size: int = 32,
                 flatten: bool = False

                 ):
        super(ImageAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size
        self.flatten = flatten 
    def forward(self, x):
        batch_size = x.shape[0]
        original_shape = x.shape  # Store the original shape
        
        if self.flatten:
            x = x.view(batch_size, -1)
        
        # Encode and decode
        latent = self.encoder(x)
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
        self.embedded_size = embedded_size  # Store embedded_size as class attribute
        self.mask_token = nn.Parameter(torch.zeros(1, embedded_size))  # Use the provided embedded_size
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

        # Reconstruct the image from patches
        reconstructed_image = torch.zeros_like(x)
        patch_size = self.patch_encoder.patch_size
        idx = 0
        
        # Calculate the expected number of elements per patch
        patch_elements = x.size(1) * patch_size * patch_size  # channels * height * width
        
        # Iterate through the image grid by patch positions
        for i in range(0, x.size(2), patch_size):
            for j in range(0, x.size(3), patch_size):
                if idx >= reconstructed_patches.size(1):
                    # Skip if we've run out of patches (can happen with rounding issues)
                    continue
                    
                # Get the current patch from the reconstructed patches
                current_patch = reconstructed_patches[:, idx, :]
                
                # Calculate shape parameters for reshaping
                batch_size = x.size(0)       # Number of images in batch
                channels = x.size(1)         # Number of channels in image
                
                # Check if the sizes match before reshaping
                if current_patch.size(1) == patch_elements:
                    # Reshape the patch from flattened representation to spatial dimensions:
                    # [batch_size, embedding] -> [batch_size, channels, patch_size, patch_size]
                    patch_reshaped = current_patch.reshape(
                        batch_size,      # Preserve batch dimension
                        channels,        # Number of channels
                        patch_size,      # Height of patch
                        patch_size       # Width of patch
                    )
                    
                    # Place the patch in the correct position in the final image
                    # Ensure we don't go out of bounds
                    h_end = min(i + patch_size, x.size(2))
                    w_end = min(j + patch_size, x.size(3))
                    reconstructed_image[:, :, i:h_end, j:w_end] = patch_reshaped[:, :, :(h_end-i), :(w_end-j)]
                else:
                    # If the sizes don't match, print debug info and try to adapt
                    print(f"Warning: Patch size mismatch. Expected {patch_elements}, got {current_patch.size(1)}")
                    # Try to use a direct reshape with the actual size we have
                    try:
                        # Calculate the patch size based on what we actually have
                        actual_patch_size = int((current_patch.size(1) / channels) ** 0.5)
                        if actual_patch_size > 0 and (actual_patch_size ** 2) * channels == current_patch.size(1):
                            patch_reshaped = current_patch.reshape(
                                batch_size,
                                channels,
                                actual_patch_size,
                                actual_patch_size
                            )
                            # Place in image using the actual patch size
                            h_end = min(i + actual_patch_size, x.size(2))
                            w_end = min(j + actual_patch_size, x.size(3))
                            reconstructed_image[:, :, i:h_end, j:w_end] = patch_reshaped[:, :, :(h_end-i), :(w_end-j)]
                        else:
                            print(f"Cannot determine appropriate patch size for {current_patch.size(1)} elements")
                    except Exception as e:
                        print(f"Error reshaping patch: {e}")
                
                # Move to the next patch index
                idx += 1

        return reconstructed_image, masked_indices

class MaskAutoEncoderCriterion(nn.Module):
    def __init__(self, reconstruction_loss_fn: nn.Module = nn.MSELoss(reduction='none'), mask_loss_weight: float = 1.0):
        """
        Initialize the MaskAutoEncoderCriterion.

        Args:
            reconstruction_loss_fn (nn.Module): The reconstruction loss function (e.g., MSELoss, L1Loss).
            mask_loss_weight (float): Weight to apply to the mask loss.
        """
        super(MaskAutoEncoderCriterion, self).__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.mask_loss_weight = mask_loss_weight

    def forward(self, reconstructed_image, original_image, masked_indices):
        """
        Compute the reconstruction loss for the masked tokens only.
        
        Args:
            reconstructed_image (torch.Tensor): The output from the autoencoder, shape (B, C, H, W).
            original_image (torch.Tensor): The ground truth image, shape (B, C, H, W).
            masked_indices (list of torch.Tensor): Indices of masked patches for each sample in the batch.

        Returns:
            torch.Tensor: The loss for the masked tokens.
        """
        batch_size, num_channels, height, width = original_image.shape
        patch_size = height // int(original_image.shape[-1] / reconstructed_image.size(1))  # Patch size based on resolution

        # Divide the images into patches
        patches_reconstructed = reconstructed_image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches_reconstructed = patches_reconstructed.contiguous().view(batch_size, num_channels, -1, patch_size, patch_size)

        patches_original = original_image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches_original = patches_original.contiguous().view(batch_size, num_channels, -1, patch_size, patch_size)

        # Initialize a mask for all patches
        mask = torch.zeros(patches_reconstructed.shape[2], dtype=torch.bool, device=reconstructed_image.device)

        # Set masked patches to True
        for idx in range(batch_size):
            mask[masked_indices[idx]] = True

        # Compute the reconstruction loss only for masked patches
        reconstruction_loss = self.reconstruction_loss_fn(
            patches_reconstructed[:, :, mask, :, :],
            patches_original[:, :, mask, :, :]
        )

        # Aggregate the loss and apply the mask weight
        loss = reconstruction_loss.mean() * self.mask_loss_weight
        return loss


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

