
import base64

import numpy as np
import torch

from prompt_to_key import convert_embedding_tensor_to_binary_key
from key_to_embedding import compute_embedding_and_latents_from_key
from key_to_embedding import convert_key_to_binary
from key_to_embedding import unpack_binary_key_into_binary_float_array
from key_to_embedding import convert_bin_key_to_float_array
from prompt_to_key import normalise_numbers



for size in range(9,243):
    print(f'checking size {size}')
    prompt_embeddings = 12.0 * torch.randn(size=(size, ), dtype=torch.float16)
    prompt_embeddings = normalise_numbers(prompt_embeddings)

    binary_key = convert_embedding_tensor_to_binary_key(prompt_embeddings,
                                        latents=None,
                                        latents_shape=None)

    key = base64.b64encode(bytes(binary_key)).decode('utf-8')


    key_bin = convert_key_to_binary(key, nb_bits_target=size * 15)
    prompt_embeddings_bin_size = 2 * size
    data_bin_size = prompt_embeddings_bin_size
    #print(f'key bin len={len(key_bin)} - data size={data_bin_size} ({prompt_embeddings_bin_size}+{latents_bin_size})')
    data =  unpack_binary_key_into_binary_float_array(key_bin, data_size=data_bin_size)
    prompt_embeds_data = convert_bin_key_to_float_array(data)
    assert(len(prompt_embeds_data) == size)

    prompt_embeds_orig_np = prompt_embeddings.flatten().detach().cpu().numpy()

    try:
        assert(np.allclose(prompt_embeds_orig_np, prompt_embeds_data))
    except AssertionError:
        print(prompt_embeds_orig_np)
        print(prompt_embeds_data)
        abs_diff = np.abs(prompt_embeds_orig_np - prompt_embeds_data)
        print(abs_diff)
        max_diff = np.max(abs_diff)
        print(max_diff)
        print(prompt_embeds_orig_np[abs_diff == max_diff])
        print(prompt_embeds_data[abs_diff == max_diff])
        raise

print('✅️ passed')



