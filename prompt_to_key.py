from diffusers import DiffusionPipeline
import torch

def compute_prompt_embedding(pipe: DiffusionPipeline,
                             prompt: str,
                             device='cuda',
                             num_images_per_prompt: int = 1,
                             guidance_scale: float = 7.5) -> torch.Tensor:
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = pipe._encode_prompt(prompt,
                                        device,
                                        num_images_per_prompt,
                                        do_classifier_free_guidance,
                                        None)
    return prompt_embeds

def convert_embedding_tensor_to_binary_key(embeddings: torch.Tensor) -> bytes:
    embeddings_data = embeddings.flatten().detach().cpu().numpy()
    pack_floats
    # TODO
    return

