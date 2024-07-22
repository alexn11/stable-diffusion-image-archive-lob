import base64

import numpy as np

from diffusers import DiffusionPipeline
import torch

from key_to_embedding import pack_float_array_into_binary_key



# smallest representable number under this 15 bits system
#  note: 0 cant be represented under the system
smallest_number = 2**-12 
# largest abs value
largest_positive_number = 8.+4+2+1+1/2+1/4+1/8+1/16+1/32+1/64+1/128


def normalise_numbers(x: torch.Tensor):
    x[torch.abs(x) < smallest_number] = smallest_number
    x[x > largest_positive_number] = largest_positive_number
    x[x < -largest_positive_number] = -largest_positive_number
    return x



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
    prompt_embeds = normalise_numbers(prompt_embeds)
    return prompt_embeds

def convert_embedding_tensor_to_binary_key(embeddings: torch.Tensor,
                                           latents: torch.Tensor | None = None,
                                           latents_shape=(1, 4, 52, 80)) -> bytes:
    embeddings_data = embeddings.flatten().detach().cpu().numpy()
    if((latents is None) and (latents_shape is None)):
        floats_data = embeddings_data
    else:
        if(latents is None):
            latents = 4. * torch.randn(size=latents_shape, dtype=torch.float16) # normally 16 so there's some data loss
        latents_data = latents.flatten().detach().cpu().numpy()
        floats_data = np.concatenate([ embeddings_data, latents_data, ])
    binary_key = pack_float_array_into_binary_key(floats_data)
    return binary_key


def generate_key_from_prompt(prompt: str,
                             pipe: DiffusionPipeline = None,
                             device: str = 'cuda',
                             num_images_per_prompt=1,
                             latents=None) -> str:
    prompt_embeddings = compute_prompt_embedding(pipe=pipe,
                                                 prompt=prompt,
                                                 device=device,
                                                 num_images_per_prompt=num_images_per_prompt,)
    assert(prompt_embeddings.shape == (2,77,768))
    prompt_embeddings = prompt_embeddings[0]
    prompt_embeddings[0, 19] = -28.078125
    prompt_embeddings[0, 681] = 33.09375
    #prompt_only_key = convert_embedding_tensor_to_binary_key(prompt_embeddings,
    #                                                         latents=None,
    #                                                         latents_shape=None)
    prompt_and_latents_key = convert_embedding_tensor_to_binary_key(prompt_embeddings,
                                                                    latents_shape=(1,4,52,80),
                                                                    latents=latents)
    key = base64.b64encode(bytes(prompt_and_latents_key)).decode('utf-8')
    #print(f'prompt={len(prompt_only_key)} - with lat={len(prompt_and_latents_key)} - encoded: {len(key)}')
    return key
