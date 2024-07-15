import torch

from prepare_model import prepare_config, prepare_model
from key_to_embedding import generate_random_base64, compute_embedding_from_key

config = prepare_config()
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

key = generate_random_base64()
array_key = compute_embedding_from_key(key)

prompt_embeds = torch.tensor(array_key, dtype=torch.float16).to(device).reshape((77,768))
assert(-1e-6 < prompt_embeds[0,19].item() + 28.078125 < 1e-6)
assert(-1e-6 < prompt_embeds[0,681].item() - 33.09375 < 1e-6)

prompt_embeds = torch.stack([prompt_embeds, prompt_embeds])
assert(prompt_embeds.shape == (2,77,768))

with torch.no_grad():
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
