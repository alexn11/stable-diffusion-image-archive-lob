

from matplotlib import pyplot
import numpy as np
import torch

from model_constants import latents_shape, num_inference_steps_nb_bits
from prepare_model import prepare_model, prepare_config
from prompt_to_key import compute_prompt_embedding
from prompt_to_key import compute_key_from_data, generate_key_from_prompt
from key_to_embedding import unpack_key


prompt = 'a dog'

config = prepare_config()
pipe = prepare_model(config['model_name'], config['dtype'], config['device'] )

prompt_embeds = compute_prompt_embedding(pipe,
                         prompt=prompt,
                         device=config['device'],
                         num_images_per_prompt=1,
                         single_embeddings=True,
                         debug=True)

prompt_embeds = prompt_embeds.detach().flatten().cpu().numpy()

key1 = compute_key_from_data(torch.tensor(prompt_embeds),
                            latents = None,
                            latents_shape=latents_shape,
                            num_inference_steps = None,
                            debug=True) 

key2 = generate_key_from_prompt(prompt,
                             pipe = pipe,
                             device = config['device'],
                             num_images_per_prompt=1,
                             latents=None,
                             latents_shape=latents_shape,
                             num_inference_steps=None,
                             debug=True)


unpacked_1 = unpack_key(key1, debug=True)
if(num_inference_steps_nb_bits > 0):
    _,embeds_1,_ = unpacked_1
else:
    embeds_1, _ = unpacked_1

unpacked_2 = unpack_key(key2, debug=True)
if(num_inference_steps_nb_bits > 0):
    _,embeds_2,_ = unpacked_2
else:
    embeds_2, _ = unpacked_2


abs_diff_1 = np.abs(prompt_embeds - embeds_1)
abs_diff_2 = np.abs(prompt_embeds - embeds_2)

print('diff 1')
print(abs_diff_1[abs_diff_1 > 0.])
print('  @')
print(np.argwhere(abs_diff_1))
print('diff 2')
print(abs_diff_2[abs_diff_2 > 0.])
print('  @')
print(np.argwhere(abs_diff_2))
























