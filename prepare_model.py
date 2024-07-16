from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch



def load_model(model_name, dtype, device) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe.to(device)
    return pipe


def prepare_config(model_name = 'stabilityai/stable-diffusion-2-1-unclip-small',
                   device = 'cuda',
                   num_inference_steps = 50,
                   prompt = '',
                   height = 416,
                   width = 640, ##
                   # no reasin to change the bellow
                   batch_size = 1,
                   num_images_per_prompt = 1,
                   guidance_scale=7.5,
                   output_type='pil'):
    guidance_scale = guidance_scale
    output_type = output_type
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
        'output_type': output_type,
        'batch_size': batch_size,
        'num_image_per_prompt': num_images_per_prompt,
        'dtype': dtype,
        'do_classifier_free_guidance': do_classifier_free_guidance,
    }

def prepare_model(model_name, dtype, device) -> DiffusionPipeline:
    pipe = load_model(model_name, dtype, device)
    return pipe

def prepare_latents(batch_size,
                    num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    dtype,
                    device,
                    vae_scale_factor):
    shape_base = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
    )
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    #return torch.randn(shape_base + (latent_height, latent_width), dtype=dtype, device=device)
    latents = torch.zeros(shape_base + (latent_height * latent_width, ), dtype=dtype, device=device)
    for channel_i in range(num_channels_latents):
        latents[:, channel_i, channel_i::3] = 1.
    return latents.reshape(shape_base + (latent_height, latent_width))