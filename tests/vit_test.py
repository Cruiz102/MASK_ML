import pytest
from mask_ml.model.vit import VitClassificationConfig, VitClassificationHead, ViTConfig, generate_all_valid_configurations
import torch
import random
def test_vit_model_with_specific_configuration():
    # Test Configuration
    ###################
    image_size = 28
    output_size = 10
    ###################
    configurations = generate_all_valid_configurations(image_size)
    conf = random.choice(configurations)
    print("Configuration used for the test:", conf)
    k = conf['kernel']
    s = conf['stride']
    att = conf['attention_heads']
    dummy_image = torch.randn(1, 3, image_size, image_size)
    model = VitClassificationHead(
        input_size=  image_size,
        num_classes=output_size,
        transformer_blocks=1,
        encoder_stride=s,
        patch_size=k,
        embedded_size=image_size,
        attention_heads=att


    )
    model.eval()
    output_probs, atten_h = model(dummy_image)
    assert output_probs.shape == (1, output_size), f"Expected output shape (1, output_size), got {output_probs.shape}"

    print("Forward pass successful with output:", output_probs)

def test_vit_model_with_multiple_configurations():
    # Define test parameters
    ###################
    image_sizes = [28, 32, 64]
    output_sizes = [10, 50, 100]
    channels = [1, 3, 5] 
    ###################

    # Iterate over each image size
    for image_size in image_sizes:
        # Generate configurations specific to the image size
        configurations = generate_all_valid_configurations(image_size)
        
        # Iterate over all configurations
        for conf in configurations:
            k = conf['kernel']
            s = conf['stride']
            att = conf['attention_heads']

            # Check if the configuration is valid
            if k > image_size or s > image_size or att <= 0:
                print(f"Skipping invalid configuration: {conf}")
                continue

            # Iterate over each number of channels
            for channel in channels:
                # Iterate over each output size
                for output_size in output_sizes:
                    print("Testing Configuration:", conf, f"Channels: {channel}, Output Size: {output_size}")

                    dummy_image = torch.randn(1, channel, image_size, image_size)
                    model = VitClassificationHead(
                        input_size= image_size,
                        num_classes=output_size,
                        transformer_blocks=1,
                        encoder_stride=s,
                        patch_size=k,
                        num_channels=channel ,
                        embedded_size=image_size,
                        attention_heads=att
                    )
                    model.eval()
                    output_probs, attn_h = model(dummy_image)
                    assert output_probs.shape == (1, output_size), f"Expected output shape (1, {output_size}), got {output_probs.shape}"
                    print("Forward pass successful with configuration:", conf)

def test_patch_encoder_interpolation():
    # Test Configuration
    ###################
    image_size = 28
    output_size = 10
    ###################
    dummy_image = torch.randn(1, 3,70, 70)
    model = VitClassificationHead(
        input_size=  image_size,
        num_classes=output_size,
        transformer_blocks=1,
        encoder_stride=4,
        patch_size=14,
        embedded_size=image_size,
        attention_heads=1,
        interpolation=True,
        interpolation_scale=1,
        num_channels=3,


    )
    model.eval()
    output_probs, attn_h = model(dummy_image)
    assert output_probs.shape == (1, output_size), f"Expected output shape (1, output_size), got {output_probs.shape}"




# Run the test with Pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])
