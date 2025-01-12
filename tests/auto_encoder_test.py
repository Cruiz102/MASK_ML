import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mask_ml.model.auto_encoder import ImageAutoEncoder
from mask_ml.model.mlp import MLP
from mask_ml.model.vit import VitModel, generate_all_valid_configurations
from utils import get_layer_output



def test_get_layer_for_latent_representation_pca():
    # Test Configuration
    ###################
    image_size = 28
    batch_size = 8
    latent_spaces = [10,20,30,32,50]
    for latent_dim in latent_spaces:    
        autoencoder = ImageAutoEncoder(
            image_size= image_size,
            flatten= False,
            image_encoder=MLP(
                activation_function="relu",
                input_size=image_size,
                output_size=latent_dim
            ),
            decoder=MLP(
                activation_function="relu",
                input_size=latent_dim,
                output_size=image_size
            )
            )
        image = torch.randn([batch_size,image_size])
        latent_representation = get_layer_output(autoencoder, image, 'image_encoder',batch_size, flatten=True) 
        assert (latent_representation.shape[0], latent_representation.shape[1]) == (batch_size, latent_dim), f"Expected latent vector to be ({batch_size}, {latent_dim}), got ({latent_representation.shape[0]}, {latent_representation[1]})"

    vit_configs = generate_all_valid_configurations(image_size)
    vit_configs = vit_configs[len(vit_configs)//2]
    vit_model = VitModel(
        transformer_blocks=1,
        image_size=image_size,
        patch_size=vit_configs["kernel"],
        num_channels= 3,
        encoder_stride=vit_configs['stride'],
        embedded_size= image_size,
        attention_heads=vit_configs['attention_heads'],
    )
    x = torch.randn([batch_size, 3, image_size, image_size])
    embedded_layer_vit = get_layer_output(vit_model, x, 'embedded_layer' )
    assert (embedded_layer_vit.shape[0], embedded_layer_vit.shape[1], embedded_layer_vit.shape[2]) == (batch_size,485,image_size), f"Expected latent vector to be ({batch_size}, {485}, {image_size}), got ({embedded_layer_vit.shape[0]}, {embedded_layer_vit.shape[1]}, {embedded_layer_vit.shape[2]})"


    print("Forward pass successful")


def test_auto_encoder_on_evaluation():
    pass
    



# Run the test with Pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])

