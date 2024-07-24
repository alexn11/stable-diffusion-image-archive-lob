
import base64
import random

import numpy as np
import torch

from model_constants import latents_shape, prompt_embeddings_shape
from model_constants import num_inference_steps_level_to_counts
from model_constants import prompt_embeddings_special_values

from key_to_embedding import unpack_key
from prompt_to_key import compute_key_from_data, compute_prompt_embedding
from prompt_to_key import normalise_numbers


def show_binary_data(array:np.ndarray, length=12):
    data = bytes(array.data)
    for i in range(length):
        print(f'{data[i]:08b}')

def show_diagnostic(orig: np.ndarray, k: np.ndarray):
    abs_diff = np.abs(orig - k)
    max_diff = np.max(abs_diff)
    argmax_diff = np.argmax(abs_diff)
    print(f'max diff={max_diff} ({argmax_diff})')
    print(f'original: - {orig[0]}')
    show_binary_data(orig)
    print(f'key: - {k[0]}')
    show_binary_data(k)


prompt_embeddings = 16.0 * torch.randn(size=prompt_embeddings_shape, dtype=torch.float16)
prompt_embeddings = normalise_numbers(prompt_embeddings, type='prompt')
prompt_embeddings_orig = prompt_embeddings.flatten().detach().cpu().numpy()
for i, v in prompt_embeddings_special_values.items():
    prompt_embeddings_orig[i] = v

latents = torch.randn(size=latents_shape, dtype=torch.float16)
latents = normalise_numbers(latents, type='latents')
latents_orig = latents.flatten().detach().cpu().numpy()

num_inference_steps = random.choice(num_inference_steps_level_to_counts)
key = compute_key_from_data(embeddings=prompt_embeddings,
                            latents=latents,
                            latents_shape=latents_shape,
                            num_inference_steps=num_inference_steps,
                            debug=True)

(
    num_inference_steps_k,
    prompt_embeddings_k,
    latents_k
) = unpack_key(key, debug=True)

assert(num_inference_steps == num_inference_steps_k)
try:
    assert(np.allclose(prompt_embeddings_orig, prompt_embeddings_k))
except AssertionError:
    show_diagnostic(prompt_embeddings_orig, prompt_embeddings_k)
    raise

try:
    assert(np.allclose(latents_orig, latents_k))
except AssertionError:
    show_diagnostic(latents_orig, latents_k)
    raise

print('✅️ passed')



