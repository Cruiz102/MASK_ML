import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.optim.adamw import AdamW
from mask_ml.utils.datasets import create_dataloader

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
        if self.flatten:
            #TENSOR_SIZE: [BATCH,CHANNEL, IMAGE_SIZE, IMAGE_SIZE] 
            x = x.flatten(2)  # Flatten -> [BATCH, CHANNEL, IMAGE_SIZE * IMAGE_SIZE]
        x = self.image_encoder(x)
        x = self.decoder(x)
        return x
    

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


import torch
import torch.nn as nn

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
        # Enable gradient computation for the inputs
        inputs.requires_grad = True
        
        # Forward pass through the encoder to get latent representation
        latent = model.image_encoder(inputs)
        
        # Compute reconstruction
        reconstruction = model(inputs)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstruction, inputs)

        # Compute the Jacobian of the latent representation
        # Sum the latent representations to aggregate gradients
        latent_sum = torch.sum(latent)
        latent_sum.backward(retain_graph=True)

        # Compute the Frobenius norm of the Jacobian for the contractive loss
        contractive_loss = 0
        for param in inputs.grad:
            contractive_loss += torch.sum(param ** 2)
        
        # Combine reconstruction and contractive losses
        loss = recon_loss + self.lambda_reg * contractive_loss

        # Reset gradients on inputs
        inputs.requires_grad = False

        return loss



class Trainer:
    def __init__(self, model , optimizer , criterion , dataloader ,checkpoint_dir, device="cuda"):
        self.model = model.to(device = device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir


    def save_model_checkpoint(self, save_path: str):
        """Save the model checkpoint."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model checkpoint saved at {save_path}")

    def train_model(self, epochs):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}")
            self.model.train()
            train_loss = 0.0
            # Training loop with tqdm
            with tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]") as pbar:
                for images, _ in pbar:
                    images = images.to(device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, images)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": loss.item()})

            train_loss /= len(self.dataloader)
            print(f"Epoch {epoch + 1}/{epochs} completed. Average Train Loss: {train_loss:.4f}")
            # Save model checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"autoencoder_epoch_{epoch + 1}.pth")
            self.save_model_checkpoint(checkpoint_path)



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


def evaluate_model(model: nn.Module, eval_loader: DataLoader, device: torch.device, output_dir: str):
    """
    Evaluate the model and save reconstructed images.

    Args:
        model (nn.Module): Trained autoencoder.
        eval_loader (DataLoader): DataLoader for evaluation.
        device (torch.device): Device to run the model on.
        output_dir (str): Directory to save reconstructed images.
    """
    # Move model to the specified device
    model.to(device)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            images = images.to(device)

            outputs = model(images)
            images_cpu = images.cpu()
            outputs_cpu = outputs.cpu()

            for i in range(images_cpu.size(0)):
                original_image = transforms.ToPILImage()(images_cpu[i])
                reconstructed_image = transforms.ToPILImage()(outputs_cpu[i])

                # Save images
                original_image.save(os.path.join(output_dir, f"original_{batch_idx * eval_loader.batch_size + i}.png"))
                reconstructed_image.save(os.path.join(output_dir, f"reconstructed_{batch_idx * eval_loader.batch_size + i}.png"))

    print(f"Reconstructed images saved to {output_dir}")





@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


    dataloader_train, dataloader_test = create_dataloader(cfg)
    model = instantiate(cfg.model)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model=model, optimizer=optimizer, dataloader= dataloader_train,
                      criterion=SparseAutoencoderCriterion(),
                      checkpoint_dir="./", device='cuda')
    
    trainer.train_model(cfg.epoch)


if __name__ == "__main__":
    # Path to COCO dataset
    coco_root = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'coco')
    train_images = f"{coco_root}/2014_train_images/train2014"
    val_images = f"{coco_root}/2014_val_images/val2014"
    annotations = f"{coco_root}/2014_train_val_annotations/annotations"
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define a transform for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert PIL images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    val_dataset = CocoDetection(
        root=val_images,
        annFile=f"{annotations}/instances_val2014.json",
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

