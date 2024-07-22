import numpy as np

from diffusers import DiffusionPipeline
import torch

from key_to_embedding import pack_float_array_into_binary_key

def compute_prompt_embedding(pipe: DiffusionPipeline,
                             prompt: str,
                             device='cuda',
                             num_images_per_prompt: int = 1,
                             guidance_scale: float = 7.5) -> torch.Tensor:
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = pipe._encode_prompt(prompt,
                                        device,
                                        num_images_per_prompt,
                                        do_classifier_free_guidance,
                                        None)
    return prompt_embeds

def convert_embedding_tensor_to_binary_key(embeddings: torch.Tensor,
                                           latents: torch.Tensor | None = None,
                                           latents_shape=(1, 4, 52, 80)) -> bytes:
    embeddings_data = embeddings.flatten().detach().cpu().numpy()
    if((latents is None) and (latents_shape is None)):
        floats_data = embeddings_data
    else:
        if(latents is None):
            latents = 16. * torch.randn(size=latents_shape, dtype=torch.float16)
        latents_data = latents.flatten().detach().cpu().numpy()
        floats_data = np.concatenate([ embeddings_data, latents_data, ])
    binary_key = pack_float_array_into_binary_key(floats_data)
    return binary_key

