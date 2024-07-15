# from diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import argparse

from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch


from prepare_model import prepare_config, prepare_model


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('prompt', type=str, default='this is the prompt')
arg_parser.add_argument('--width', type=int, default=640)
arg_parser.add_argument('--height', type=int, default=416)
arg_parser.add_argument('--device', type=str, choices=['cuda', 'cpu', ], default='cuda')
arg_parser.add_argument('--num-inference-steps', type=int, default=50)
arg_parser.add_argument('--model-name', type=str, default='stabilityai/stable-diffusion-2-1-unclip-small')
parsed_args = arg_parser.parse_args()

model_name = parsed_args.model_name
device = parsed_args.device
num_inference_steps = parsed_args.num_inference_steps
prompt = parsed_args.prompt
height = parsed_args.height
width = parsed_args.width

config = prepare_config(model_name=model_name,
                        device=device,
                        num_inference_steps=num_inference_steps,
                        prompt=prompt,
                        height=height,
                        width=width)
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
pipe = prepare_model(model_name, dtype, device)



def embed_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance):
    prompt_embeds = pipe._encode_prompt(prompt,
                                        device,
                                        num_images_per_prompt,
                                        do_classifier_free_guidance,
                                        None)
    return prompt_embeds


with torch.no_grad():
    prompt_embeds = embed_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance)
    #
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    #
    num_channels_latents = pipe.unet.config.in_channels
    #
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator=None,
        latents=None,
    )
    #
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
    #
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
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
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


image[0].show()

"""
if not return_dict:
    return (image, has_nsfw_concept)

return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
"""





"""




        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]


        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

        

"""