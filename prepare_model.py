# Copyright 2024 Alexandre De Zotti. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch

from model_constants import image_height, image_width


def load_model(model_name, dtype, device) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def prepare_config(model_name = 'stabilityai/stable-diffusion-2-1-unclip-small',
                   device = 'cuda',
                   num_inference_steps = 50,
                   prompt = '',
                   # no reason to change anything bellow
                   height = image_height,
                   width = image_width,
                   batch_size = 1,
                   num_images_per_prompt = 1,
                   guidance_scale=7.5,):
    guidance_scale = guidance_scale
    batch_size = batch_size # = number of prompts
    num_images_per_prompt = num_images_per_prompt
    dtype = torch.float32 if(device == 'cpu') else torch.float16
    do_classifier_free_guidance = guidance_scale > 1.0
    return {
        'model_name': model_name,
        'device': device,
        'num_inference_steps': num_inference_steps,
        'prompt': prompt,
        'height': height,
        'width': width,
        'guidance_scale': guidance_scale,
        'batch_size': batch_size,
        'num_images_per_prompt': num_images_per_prompt,
        'dtype': dtype,
        'do_classifier_free_guidance': do_classifier_free_guidance,
    }

def prepare_model(model_name, dtype, device) -> DiffusionPipeline:
    pipe = load_model(model_name, dtype, device)
    return pipe

def add_special_filter_to_base_latents(latents, width, height, device, dtype):
    x = torch.linspace(-width, width, width, dtype=dtype, device=device, requires_grad=False)
    y = torch.linspace(-height, height, height, dtype=dtype, device=device, requires_grad=False)
    x_grid, y_grid = torch.meshgrid(y, x)
    normalisation_factor = 2. / (0.5 * (width + height)**2)
    distance_to_center = ((x_grid*x_grid) + (y_grid*y_grid)) * normalisation_factor
    print(f'lat={latents.shape}, dis={distance_to_center.shape}')
    latents = torch.exp(-distance_to_center) * latents
    return latents

def prepare_latents(batch_size,
                    num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    dtype,
                    device,
                    vae_scale_factor,
                    generator=None,
                    debug=False):
    shape_base = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
    )
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    if(debug):
        print(f'latents shape: {shape_base + (latent_height, latent_width)}')
    if(generator is not None):
        latents = torch.randn(shape_base + (latent_height, latent_width),
                              dtype=dtype,
                              device=device,
                              generator=generator)
    else:
        latents = torch.zeros(shape_base + (latent_height * latent_width, ), dtype=dtype, device=device)
        for channel_i in range(num_channels_latents):
            latents[:, channel_i, channel_i::3] = 1.
        latents = latents.reshape(shape_base + (latent_height, latent_width))
        latents = add_special_filter_to_base_latents(latents,
                                                    width=latent_width,
                                                    height=latent_height,
                                                    device=device,
                                                    dtype=dtype)
    return latents