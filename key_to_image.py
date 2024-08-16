import argparse

import numpy as np
import torch
from diffusers.utils.testing_utils import enable_full_determinism

from model_constants import data_nb_bits
from model_constants import num_inference_steps_nb_bits
from model_constants import prompt_embeddings_shape, latents_shape
from prepare_model import prepare_config, prepare_model
from key_strings import generate_random_key_base64
from prompt_to_key import generate_key_from_prompt, compute_key_from_data
from key_to_image_pipeline import key_to_image

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prompt', type=str, default='')
arg_parser.add_argument('--embeddings-file', type=str, default='')
#arg_parser.add_argument('--width', type=int, default=640)
#arg_parser.add_argument('--height', type=int, default=416)
arg_parser.add_argument('--device', type=str, choices=['cuda', 'cpu', ], default='cuda')
arg_parser.add_argument('--num-inference-steps', type=int, default=50)
arg_parser.add_argument('--model-name', type=str, default='stabilityai/stable-diffusion-2-1-unclip-small')
arg_parser.add_argument('--nb-keys', type=int, default=8)
arg_parser.add_argument('--key-file', type=str, default='')
arg_parser.add_argument('--seed', type=int, default=768)
arg_parser.add_argument('--latents-seed', type=int, default=-33)
arg_parser.add_argument('--check-determinism', action='store_true')
#arg_parser.add_argument('--latents-type', type=str, choices=['blob', 'fixed-generator']) # obsolete
arg_parser.add_argument('--output-file-name', type=str, default='')
arg_parser.add_argument('--no-debug', action='store_true')
arg_parser.add_argument('--show-latents', action='store_true')
#arg_parser.add_argument('--skip', type=int, default=0)
parsed_args = arg_parser.parse_args()

config_dict = parsed_args.__dict__.copy()
del(config_dict['nb_keys'])
del(config_dict['key_file'])
del(config_dict['seed'])
del(config_dict['check_determinism'])
#del(config_dict['latents_type'])
del(config_dict['latents_seed'])
del(config_dict['output_file_name'])
del(config_dict['no_debug'])
del(config_dict['show_latents'])
del(config_dict['embeddings_file'])
#del(config_dict['skip'])

config = prepare_config(**config_dict)
model_name = config['model_name']
device = config['device']
if(num_inference_steps_nb_bits == 0):
    num_inference_steps = config['num_inference_steps']
else:
    num_inference_steps = None
prompt = config['prompt']
height = config['height']
width = config['width']
guidance_scale = config['guidance_scale']
batch_size = config['batch_size']
num_images_per_prompt = config['num_images_per_prompt']
dtype = config['dtype']
do_classifier_free_guidance = config['do_classifier_free_guidance']
embeddings_file = parsed_args.embeddings_file

nb_keys = parsed_args.nb_keys
key_file_path = parsed_args.key_file
#key_step_skip = parsed_args.skip

do_show_latents = parsed_args.show_latents
do_debug = not parsed_args.no_debug

if(prompt != ''):
    print(f'using key generated from prompt "{prompt}"')
    keys = []
elif(embeddings_file != ''):
    print(f'using keys generated from synthetic embeddings ({embeddings_file})')
    keys = []
elif(key_file_path != ''):
    print(f'using key from file "{key_file_path}"')
    with open(key_file_path, 'r') as key_file:
        key = key_file.read().strip()
    keys =  [ key, ]
    if(parsed_args.check_determinism):
        keys = [ key, key, key ]
else:
    print(f'generating {nb_keys} keys')
    keys = [ generate_random_key_base64(data_nb_bits,) for i in range(nb_keys) ]
    #for k in keys:
    #    print(f'{len(k)}')
#array_key = compute_embedding_from_key(key)
#prompt_embeds = torch.tensor(array_key, dtype=torch.float16).to(device).reshape((77,768))
#assert(-1e-6 < prompt_embeds[0,19].item() + 28.078125 < 1e-6)
#assert(-1e-6 < prompt_embeds[0,681].item() - 33.09375 < 1e-6)
#prompt_embeds = torch.stack([prompt_embeds, prompt_embeds])
#assert(prompt_embeds.shape == (2,77,768))


enable_full_determinism()


generator = None

pipe_generator = torch.Generator(device=device).manual_seed(parsed_args.seed)
#pipe_generator = None
pipe = prepare_model(model_name, dtype, device, )

if(prompt != ''):
    print('computing keys for prompt')
    keys = [
            generate_key_from_prompt(prompt=prompt,
                                    pipe=pipe,
                                    device=device,
                                    num_images_per_prompt=num_images_per_prompt,
                                    latents=None,)
            for k in range(nb_keys)
        ]
    #key_0 = generate_key_from_prompt(prompt=prompt,
    #                                 pipe=pipe,
    #                                 device=device,
    #                                 num_images_per_prompt=num_images_per_prompt,
    #                                 latents=None,)
    #nb_steps = nb_keys
    #step_size = 2
    #keys = [ key_0 ]
    #prev_key = key_0
    #for step_i in range(nb_steps):
    #    next_key = get_next_key(prev_key)
    #    #for i in range(1845):
    #    for i in range(step_size):
    #        #next_key = get_next_key(next_key, skip=0)
    #        next_key = get_next_key(prev_key, skip=key_step_skip)
    #    keys.append(next_key)
    #    prev_key = next_key
    #print(f'k:{keys[0] == keys[1]}')
    #for i in range(len(key_0)):
    #    a = keys[0][i]
    #    b = keys[1][i]
    #    if(a != b):
    #        print(f'd={i}: "{a}" - "{b}" -- i={i}+{key_step_skip}={i+key_step_skip} - len={len(key_0)}')
    #raise Exception('end')
    #keys += [ get_next_key(key) for key in keys[:nb_keys] ]
    #keys += [ get_next_key(key, direction=-1) for key in keys[:nb_keys] ]
elif(embeddings_file != ''):
    embeddings = np.load(embeddings_file)
    nb_embeddings = len(embeddings)
    embeddings = torch.tensor(embeddings).reshape((nb_embeddings, 2, 77, 768)).to(torch.float16).to(device)
    keys = [
                compute_key_from_data(embeddings=embedding[1],
                                latents=None,
                                latents_shape=latents_shape,
                                num_inference_steps=num_inference_steps,
                                debug=do_debug)
                for embedding in embeddings
            ]

"""
vae_scale_factor = pipe.vae_scale_factor
num_channels = pipe.unet.config.in_channels
print(f'💄️ vae scale factor: {vae_scale_factor}')
print(f'💄️ num channels: {num_channels}')
latents = prepare_latents(batch_size=batch_size,
                          num_channels_latents=num_channels,
                          num_images_per_prompt=num_images_per_prompt,
                          device=device,
                          dtype=dtype,
                          height=height,
                          width=width,
                          vae_scale_factor=vae_scale_factor,
                          generator=generator)
"""
      
for key_i, key in enumerate(keys):
    print(f'key len={len(key)}')
    print(f' {key[189434:]}')
    image = key_to_image(key=key,
                         pipe=pipe,
                         generator=pipe_generator,
                         dtype=dtype,
                         device=device,
                         prompt_embeddings_shape=prompt_embeddings_shape,
                         latents_shape=latents_shape,
                         num_inference_steps=num_inference_steps,
                         num_images_per_prompt=num_images_per_prompt,
                         batch_size=batch_size,
                         guidance_scale=guidance_scale,
                         do_classifier_free_guidance=do_classifier_free_guidance,
                         debug=do_debug)
    print(image)
    print(type(image))
    image.show()
    if(parsed_args.output_file_name != ''):
        output_file_path = f'{parsed_args.output_file_name}-{key_i:03d}.png'
        image.save(output_file_path)
