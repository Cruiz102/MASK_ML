import pytest
from mask_ml.model.vit import VitClassificationConfig, VitClassificationHead, ViTConfig, generate_all_valid_configurations
from torchvision import transforms
import torchvision
import torch

def test_vit_model_with_specific_configuration():
    # Generate the 7th-to-last configuration
    image_size = 28
    conf = generate_all_valid_configurations(image_size)[-2]
    print("Configuration used for the test:", conf)
    # Extract kernel, stride, attention_heads, and sequence_per_image
    k = conf['kernel']
    s = conf['stride']
    att = conf['attention_heads']

    # Define a ViT classification configuration
    vit_classification = VitClassificationConfig(
        model_config=ViTConfig(
            image_size=image_size,
            transformer_blocks=1,
            encoder_stride=s,
            patch_size=k,
            embedded_size=image_size,  # Example embedded size
            attention_heads=att
        ),
        input_size=image_size,  # Example input size
        num_classes=10  # Example number of classes
    )

    # Create a dummy image tensor
    dummy_image = torch.randn(1, 3, vit_classification.model_config.image_size, vit_classification.model_config.image_size)

    # Initialize the ViT classification model
    model = VitClassificationHead(vit_classification)
    model.eval()

    # Perform a forward pass with the dummy image
    output = model(dummy_image)

    # Validate the model's output shape
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

    print("Forward pass successful with output:", output)

# Run the test with Pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])
