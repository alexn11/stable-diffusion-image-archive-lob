
from diffusers import DiffusionPipeline
from PIL.Image import Image
import torch

from model_constants import prompt_embeddings_shape, latents_shape
from model_constants import num_inference_steps_nb_bits
from model_constants import image_height, image_width
from key_to_embedding import unpack_key
from prepare_model import prepare_latents




def key_to_image(key: str,
                 pipe: DiffusionPipeline,
                 generator: torch.Generator = None,
                 dtype = torch.float16,
                 device = 'cuda',
                 num_inference_steps: int | None = None,
                 num_images_per_prompt: int = 1,
                 batch_size: int = 1,
                 height: int = image_height,
                 width: int = image_width,
                 prompt_embeddings_shape: tuple = prompt_embeddings_shape,
                 latents_shape: tuple = latents_shape,
                 guidance_scale: float = 7.5,
                 do_classifier_free_guidance: bool = False,
                 max_steps: int = 64,
                 debug=False) -> Image:
    output_type = 'pil'
    if(num_inference_steps_nb_bits > 0):
        (
            num_inference_steps,
            prompt_embeds_data,
            latents_data
        ) = unpack_key(key, debug=debug)
    else:
        (
            prompt_embeds_data,
            latents_data
        ) = unpack_key(key, debug=debug)
    # forces saving on compute:
    if((max_steps > 0) and (num_inference_steps > max_steps)):
        num_inference_steps = max_steps
    prompt_embeds = torch.tensor(prompt_embeds_data, dtype=dtype).to(device).reshape(prompt_embeddings_shape)
    prompt_embeds = torch.stack([prompt_embeds, prompt_embeds])
    seed_image = torch.tensor(latents_data, dtype=dtype).reshape(latents_shape)
    # >👲️👲️👲️
    #seed_image = prepare_latents(batch_size=batch_size,
    #                             num_images_per_prompt=num_images_per_prompt,
    #                             num_channels_latents= pipe.unet.config.in_channels,
    #                             height=height,
    #                             width=width,
    #                             dtype=dtype,
    #                             device=device,
    #                             vae_scale_factor=pipe.vae_scale_factor,
    #                             debug=True)
    # <👲️👲️👲️
    #
    # >👲️👲️👲️
    #print(seed_image.shape)
    #seed_image = torch.randn(size=latents_shape,dtype=torch.float16).to('cuda')
    #print(seed_image.shape)
    #if(False):
    #    prompt_embeds2 = pipe._encode_prompt(prompt,
    #                                        device,
    #                                        num_images_per_prompt,
    #                                        do_classifier_free_guidance,
    #                                        None)
    #    print(f'👲️👲️ prepared={prompt_embeds[0,0,19]} - {prompt_embeds[0,0,681]}')
    #    print(f'👲️👲️ computed={prompt_embeds2[0,0,19]} - {prompt_embeds2[0,0,681]}')
    #    abs_diff = torch.abs(prompt_embeds - prompt_embeds2)
    #    print(f'where diff: {torch.argmax(abs_diff[0])} - {torch.max(abs_diff[0]).item()}')
    #    #from matplotlib import pyplot
    #    #pyplot.plot(abs_diff.flatten().detach().cpu().numpy())
    #    #pyplot.show()
    #    # 👲️👲️ 
    #    prompt_embeds[1,:] = prompt_embeds2[1, :]
    #    # 👲️👲️ 
    #    #prompt_embeds = prompt_embeds2
    #raise Exception('end')
    # <👲️👲️👲️
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
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        #
        do_denormalize = [True] * image.shape[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        #
        # Offload last model to CPU
        if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
            pipe.final_offload_hook.offload()
    return image[0]
