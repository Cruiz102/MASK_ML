import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm
import os

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Adaptive pooling to reduce image size
        self.pooling_layer = nn.AdaptiveAvgPool2d((32, 32))  # Downsample to 32x32
        self.flatten_size = 32 * 32 * 3  # Flattened size after pooling (32x32 RGB)

        # Encoder and decoder layers
        self.encoder_linear_layers = [self.flatten_size, 1024, 256]
        self.decoder_linear_layers = [256, 1024, self.flatten_size]

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [nn.Linear(self.encoder_linear_layers[i - 1] if i > 0 else self.flatten_size, width)
             for i, width in enumerate(self.encoder_linear_layers)]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [nn.Linear(self.decoder_linear_layers[i - 1], width) if i > 0 else nn.Linear(self.encoder_linear_layers[-1], width)
             for i, width in enumerate(self.decoder_linear_layers)]
        )

        # Upsampling layer to match input dimensions
        self.upsample_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        self.activation = nn.ReLU()

    def forward(self, x):
        # Adaptive pooling to reduce size
        x = self.pooling_layer(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Encode
        for layer in self.encoder_layers:
            x = self.activation(layer(x))

        # Decode
        for layer in self.decoder_layers:
            x = self.activation(layer(x))

        # Reshape back to pooled image size
        x = x.view(x.size(0), 3, 32, 32)

        # Upsample to match input size
        x = self.upsample_layer(x)

        return x


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

def save_model_checkpoint(model: nn.Module, save_path: str):
    """Save the model checkpoint."""
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved at {save_path}")

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
            # Move images to the device
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Move outputs to CPU for saving
            images_cpu = images.cpu()
            outputs_cpu = outputs.cpu()

            # Save original and reconstructed images
            for i in range(images_cpu.size(0)):
                # If you applied normalization during preprocessing, you may need to unnormalize here
                # For example:
                # unnormalize = transforms.Normalize(
                #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                #     std=[1/0.229, 1/0.224, 1/0.225]
                # )
                # img = unnormalize(images_cpu[i])

                # Convert tensors to PIL Images
                original_image = transforms.ToPILImage()(images_cpu[i])
                reconstructed_image = transforms.ToPILImage()(outputs_cpu[i])

                # Save images
                original_image.save(os.path.join(output_dir, f"original_{batch_idx * eval_loader.batch_size + i}.png"))
                reconstructed_image.save(os.path.join(output_dir, f"reconstructed_{batch_idx * eval_loader.batch_size + i}.png"))

    print(f"Reconstructed images saved to {output_dir}")

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, checkpoint_dir: str):
    """
    Train the autoencoder and save model checkpoints.
    """
    lr = 2e-5
    epochs = 10
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Using MSELoss for reconstruction error

    # Move model to GPU
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        # Training loop with tqdm
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]") as pbar:
            for images, _ in pbar:
                images = images.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, images)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Train Loss: {train_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"autoencoder_epoch_{epoch + 1}.pth")
        save_model_checkpoint(model, checkpoint_path)

        # Validation phase with tqdm
        model.eval()
        val_loss = 0.0
        print("Starting validation phase...")

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]") as pbar:
                for images, _ in pbar:
                    images = images.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    val_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": loss.item()})

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Validation Loss: {val_loss:.4f}")


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

    # Initialize the model
    model = AutoEncoder()

    # Train the model
    checkpoint_dir = "./checkpoints"
    train_model(model, val_loader, val_loader, device, checkpoint_dir)

    # Evaluate the model and save reconstructed images
    eval_output_dir = "./reconstructed_images"
    evaluate_model(model, val_loader, device, eval_output_dir)
