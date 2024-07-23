import argparse
import base64
from functools import reduce

import torch
from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism

from model_constants import data_nb_bits
from model_constants import prompt_embeddings_shape, latents_shape
from prepare_model import prepare_config, prepare_model
from key_strings import generate_random_key_base64
from key_to_embedding import unpack_key
from prompt_to_key import generate_key_from_prompt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prompt', type=str, default='')
#arg_parser.add_argument('--width', type=int, default=640)
#arg_parser.add_argument('--height', type=int, default=416)
arg_parser.add_argument('--device', type=str, choices=['cuda', 'cpu', ], default='cuda')
#arg_parser.add_argument('--num-inference-steps', type=int, default=50)
arg_parser.add_argument('--model-name', type=str, default='stabilityai/stable-diffusion-2-1-unclip-small')
arg_parser.add_argument('--nb-keys', type=int, default=8)
arg_parser.add_argument('--key-file', type=str, default='')
arg_parser.add_argument('--seed', type=int, default=768)
arg_parser.add_argument('--latents-seed', type=int, default=-33)
arg_parser.add_argument('--check-determinism', action='store_true')
#arg_parser.add_argument('--latents-type', type=str, choices=['blob', 'fixed-generator']) # obsolete
arg_parser.add_argument('--output-file-name', type=str, default='')
arg_parser.add_argument('--debug', action='store_true')
parsed_args = arg_parser.parse_args()

config_dict = parsed_args.__dict__.copy()
del(config_dict['nb_keys'])
del(config_dict['key_file'])
del(config_dict['seed'])
del(config_dict['check_determinism'])
#del(config_dict['latents_type'])
del(config_dict['latents_seed'])
del(config_dict['output_file_name'])
del(config_dict['debug'])

config = prepare_config(**config_dict)
model_name = config['model_name']
device = config['device']
num_inference_steps = config['num_inference_steps']
prompt = config['prompt']
height = config['height']
width = config['width']
guidance_scale = config['guidance_scale']
output_type = config['output_type']
batch_size = config['batch_size']
num_images_per_prompt = config['num_image_per_prompt']
dtype = config['dtype']
do_classifier_free_guidance = config['do_classifier_free_guidance']


nb_keys = parsed_args.nb_keys
key_file_path = parsed_args.key_file

do_debug = parsed_args.debug

if(prompt != ''):
    print(f'using key generated from prompt "{prompt}"')
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

def key_to_image(key: str,
                 pipe: DiffusionPipeline,
                 generator: torch.Generator = None,
                 dtype = torch.float16,
                 device = 'cuda',
                 prompt_embeddings_shape=prompt_embeddings_shape,
                 latents_shape=latents_shape,
                 debug=False):
    (
        num_inference_steps,
        prompt_embeds_data,
        latents_data
    ) = unpack_key(key, debug=debug)
    prompt_embeds = torch.tensor(prompt_embeds_data, dtype=dtype).to(device).reshape(prompt_embeddings_shape)
    prompt_embeds = torch.stack([prompt_embeds, prompt_embeds])
    seed_image = torch.tensor(latents_data, dtype=dtype).reshape(latents_shape)
    num_channels_latents = pipe.unet.config.in_channels

    if(debug):
        assert(latents_shape[1] == num_channels_latents)

    with torch.no_grad():
        generator = torch.Generator(device=pipe.device).manual_seed(generator.initial_seed())
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        #print(f'before: {seed_image[0][:][0][0]}')
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=None,
            latents=seed_image,
        )
        #print(f'after: {seed_image[0][:][0][0]}')
        #
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
        #
        # oh this is copy paste from diffusers... TODO: rewrite
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        #
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred,
                                              t,
                                              latents,
                                              **extra_step_kwargs,
                                              return_dict=False)[0]
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                    #if callback is not None and i % callback_steps == 0:
                    #    callback(i, t, latents)
        #
        #
        if not output_type == "latent":
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        #
        do_denormalize = [True] * image.shape[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        #
        # Offload last model to CPU
        if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
            pipe.final_offload_hook.offload()
    return image


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
    #print(f'key len={len(key)}')
    image = key_to_image(key=key,
                         pipe=pipe,
                         generator=pipe_generator,
                         dtype=dtype,
                         device=device,
                         prompt_embeddings_shape=prompt_embeddings_shape,
                         latents_shape=latents_shape,
                         debug=do_debug)
    image[0].show()
    if(parsed_args.output_file_name != ''):
        output_file_path = f'{parsed_args.output_file_name}-{key_i:03d}.png'
        image[0].save(output_file_path)
