import random

from matplotlib import pyplot
import numpy as np
import torch

from model_constants import latents_shape, num_inference_steps_nb_bits
from prepare_model import prepare_model, prepare_config
from prompt_to_key import compute_prompt_embedding
from prompt_to_key import compute_key_from_data, generate_key_from_prompt
from key_to_embedding import unpack_key

prompts = [
    'a dog',
    'vaccination',
    'a blue sky',
    'it\'s rainning today',
    'microships under skin',
    "You are wasting your money.",
    "World War 1 & 2",
    "Dropping vaccines from the sky.",
]

random.shuffle(prompts)

config = prepare_config()
pipe = prepare_model(config['model_name'], config['dtype'], config['device'] )

special_values = {
    19: [],
    681: [],
}

for prompt in prompts:
    with torch.no_grad():
        prompt_embeds = compute_prompt_embedding(pipe,
                                prompt=prompt,
                                device=config['device'],
                                num_images_per_prompt=1,
                                single_embeddings=True,
                                debug=True)
    special_values[19].append(prompt_embeds[0,19].item())
    special_values[681].append(prompt_embeds[0,681].item())


print(special_values)
















