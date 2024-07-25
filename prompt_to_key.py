import random

from diffusers import DiffusionPipeline
import torch

from model_constants import latents_shape, num_inference_steps_level_to_counts
from model_constants import prompt_embeddings_exponent_max
from FloatPacker import FloatPacker
from key_to_embedding import pack_data_into_key




def compute_prompt_embedding(pipe: DiffusionPipeline,
                             prompt: str,
                             device='cuda',
                             num_images_per_prompt: int = 1,
                             guidance_scale: float = 7.5,
                             single_embeddings=True,
                             debug=False) -> torch.Tensor:
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = pipe._encode_prompt(prompt,
                                        device,
                                        num_images_per_prompt,
                                        do_classifier_free_guidance,
                                        None)
    if(single_embeddings):
        prompt_embeds = prompt_embeds[0]
    value_normaliser = FloatPacker(max_exponent=prompt_embeddings_exponent_max,
                                   debug=debug)
    prompt_embeds = value_normaliser.normalise_numbers(prompt_embeds)
    return prompt_embeds

def compute_key_from_data(embeddings: torch.Tensor,
                          latents: torch.Tensor | None = None,
                          latents_shape=latents_shape,
                          num_inference_steps: int | None = None,
                          debug=False) -> str:
    embeddings_data = embeddings.flatten().detach().cpu().numpy()
    if((latents is None) and (latents_shape is None)):
        raise ValueError(f'obsolete undocumented functionality reached')
    else:
        if(latents is None):
            latents = torch.randn(size=latents_shape, dtype=torch.float16)
        latents_data = latents.flatten().detach().cpu().numpy()
    if(num_inference_steps is None):
        num_inference_steps = random.choice(num_inference_steps_level_to_counts)
    if(debug):
        assert(num_inference_steps in num_inference_steps_level_to_counts)
    key = pack_data_into_key(num_inference_steps=num_inference_steps,
                             prompt_embeddings=embeddings_data,
                             latents=latents_data,
                             debug=debug)
    return key


def generate_key_from_prompt(prompt: str,
                             pipe: DiffusionPipeline = None,
                             device: str = 'cuda',
                             num_images_per_prompt=1,
                             latents=None,
                             latents_shape=latents_shape,
                             num_inference_steps=None,
                             debug=False) -> str:
    if(num_inference_steps is None):
        num_inference_steps = random.choice(num_inference_steps_level_to_counts)
    prompt_embeddings = compute_prompt_embedding(pipe=pipe,
                                                 prompt=prompt,
                                                 device=device,
                                                 num_images_per_prompt=num_images_per_prompt,)
    if(debug):
        try:
            assert(prompt_embeddings.shape == (2,77,768))
        except AssertionError:
            print(f'prompt emb shape: {prompt_embeddings.shape} (expect: 2,77,768)')
            raise
    key = compute_key_from_data(embeddings=prompt_embeddings,
                                latents=latents,
                                latents_shape=latents_shape,
                                num_inference_steps=num_inference_steps,
                                debug=debug)
    return key
