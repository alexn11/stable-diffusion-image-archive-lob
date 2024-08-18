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


import struct

import numpy as np

from BitStream import BitStream
from FloatPacker import FloatPacker
from model_constants import data_nb_bits
from model_constants import nb_padding_chars
from model_constants import num_inference_steps_nb_bits
from model_constants import num_inference_steps_level_to_counts, num_inference_steps_counts_to_level
from model_constants import prompt_embeddings_bits_per_value, latents_bits_per_value
from model_constants import prompt_embeddings_nb_bytes, latents_nb_bytes
from model_constants import prompt_embeddings_exponent_max, latents_exponent_max
from model_constants import prompt_embeddings_nb_values, latents_nb_values
from model_constants import prompt_embeddings_special_values
from key_strings import convert_key_to_bit_stream, convert_packed_data_to_key


def unpack_num_inference_steps(data_stream: BitStream) -> int:
    data_stream.set_chunk_size(num_inference_steps_nb_bits)
    num_inference_steps_level = data_stream.get_chunk()
    num_inference_steps = num_inference_steps_level_to_counts[num_inference_steps_level]
    return num_inference_steps

def unpack_array(data_stream: BitStream,
                 array_type='prompt',
                 size_nb_values=None,
                 size_bytes=None,
                 debug=False) -> np.ndarray:
    if(debug):
        print_ct=0
    float_unpacker = FloatPacker(max_exponent = prompt_embeddings_exponent_max
                                                   if(array_type == 'prompt') else
                                                latents_exponent_max,
                                debug=debug)
    chunk_size = prompt_embeddings_bits_per_value if(array_type == 'prompt') else latents_bits_per_value
    if(chunk_size != 15):
        raise ValueError(f'unsupported chunk bit size: {chunk_size}')
    if(size_nb_values is None):
        size_nb_values = prompt_embeddings_nb_values if(array_type == 'prompt') else latents_nb_values
    if(size_bytes is None):
        size_bytes = prompt_embeddings_nb_bytes if(array_type == 'prompt') else latents_nb_bytes
    data_stream.set_chunk_size(chunk_size)
    packed_data = data_stream.get_chunks(size_nb_values)
    unpacked_bytes = bytearray(size_bytes * [0])
    for value_i, packed_value in enumerate(packed_data):
        unpacked_value = float_unpacker.unpack(packed_value)
        if(debug):
            if(print_ct < 24):
                print(f'unpacked_value={unpacked_value:016b}')
                print_ct += 1
        unpacked_bytes[2 * value_i + 1] = (unpacked_value >> 8) & 0xff
        unpacked_bytes[2 * value_i] = unpacked_value & 0xff
    array_as_list = list(struct.unpack(f'<{size_nb_values}e', bytes(unpacked_bytes)))
    if(debug):
        print(f'1st unpacked value (1): {array_as_list[0]}')
    if(debug):
        print(f'len read={len(array_as_list)}')
    if(array_type == 'prompt'):
        # insert special values
        array_as_list = (
            array_as_list[:19]
              + [ -28.078125, ]
              + array_as_list[19:680]
              + [ 33.09375, ]
              + array_as_list[680:])
        if(debug):
            try:
                assert(len(array_as_list) == 77*768)
            except AssertionError:
                print(f'array len={len(array_as_list)} - expected nb values: {size_nb_values}')
                raise
    if(debug):
        print(f'1st unpacked value (2): {array_as_list[0]}')
    array = np.array(array_as_list, dtype=np.float16)
    if(debug):
        print(f'1st unpacked value (3) - array: {array[0]}')
    return np.array(array)

def unpack_prompt_embeddings(data_stream: BitStream, debug=False) -> np.ndarray:
    return unpack_array(data_stream,
                        array_type='prompt',
                        debug=debug)

def unpack_latents(data_stream: BitStream, debug=False) -> np.ndarray:
    return unpack_array(data_stream,
                        array_type='latents',
                        debug=debug)

def unpack_key(base_64_key: str,
               debug=False, # i would use a logger if setting up the mode you want wasnt so complicated
               ) -> tuple[int, np.ndarray, np.ndarray]:
    data_stream = convert_key_to_bit_stream(base_64_key,
                                            start_chunk_size_bits=2,
                                            data_size_bits=data_nb_bits,
                                            nb_padding_chars=nb_padding_chars,
                                            debug=debug)
    if(num_inference_steps_nb_bits > 0):
        num_inference_steps = unpack_num_inference_steps(data_stream,)
    else:
        num_inference_steps = None
    prompt_embeddings = unpack_prompt_embeddings(data_stream,
                                                 debug=debug)
    latents = unpack_latents(data_stream, debug=debug)
    if(num_inference_steps_nb_bits > 0):
        return num_inference_steps, prompt_embeddings, latents
    return prompt_embeddings, latents

def pack_num_inference_steps(packed_data_stream: BitStream, num_inference_steps: int) -> BitStream:
    packed_data_stream.set_chunk_size(num_inference_steps_nb_bits)
    packed_data_stream.write_chunk(num_inference_steps_counts_to_level[num_inference_steps])
    return packed_data_stream

def pack_array(packed_data_stream: BitStream, array: np.ndarray, array_type='', debug = False) -> BitStream:
    assert(array_type in ['prompt', 'latents'])
    chunk_size_bits = prompt_embeddings_bits_per_value if(array_type == 'prompt') else latents_bits_per_value
    float_packer = FloatPacker(max_exponent = prompt_embeddings_exponent_max
                                                if(array_type == 'prompt') else
                                              latents_exponent_max,
                               debug=debug)
    if(chunk_size_bits != 15):
        raise ValueError(f'unsupported chunk bit size: {chunk_size_bits}')
    packed_data_stream.set_chunk_size(chunk_size_bits)
    if(array_type == 'prompt'):
        # remove special values
        if(debug):
            print(f'🎀️ 🎀️ bfore ({len(array)}) {array[19]} - {array[681]}')
        array = np.delete(array, list(prompt_embeddings_special_values.keys()))
        if(debug):
            print(f'🎀️ 🎀️ after ({len(array)}) {array[19]} - {array[681]}')
    array_len = len(array)
    if(debug):
        if(array_type == 'prompt'):
            try:
                assert(array_len == prompt_embeddings_nb_values)
            except AssertionError:
                print(f'array len={len(array)} - prompt emb nb vals={prompt_embeddings_nb_values}')
                raise
        else:
            try:
                assert(array_len == latents_nb_values)
            except AssertionError:
                print(f'array len={len(array)} - latents nb vals={latents_nb_values}')
                raise
    array_data = struct.unpack(f'{array_len}h', bytes(array.data))
    for value in array_data:
        packed_value = float_packer.pack(value)
        packed_data_stream.write_chunk(packed_value)
    return packed_data_stream

def pack_prompt_embeddings(packed_data_stream: BitStream, prompt_embeddings: np.ndarray, debug = False) -> BitStream:
    # remove special values
    return pack_array(packed_data_stream, prompt_embeddings, array_type='prompt', debug=debug)

def pack_prompt_latents(packed_data_stream: BitStream, latents: np.ndarray, debug = False) -> BitStream:
    return pack_array(packed_data_stream, latents, array_type='latents', debug=debug)

def pack_data_into_key(num_inference_steps: int | None = None,
                       prompt_embeddings: np.ndarray = None,
                       latents: np.ndarray = None,
                       debug = False,
                       return_type='key') -> str:
    packed_data_stream = BitStream(data_size_bits=data_nb_bits, mode='w')
    if(num_inference_steps_nb_bits > 0):
        pack_num_inference_steps(packed_data_stream, num_inference_steps)
    pack_prompt_embeddings(packed_data_stream, prompt_embeddings, debug=debug)
    pack_prompt_latents(packed_data_stream, latents, debug=debug)
    if(return_type != 'key'):
        return packed_data_stream
    return convert_packed_data_to_key(bytes(packed_data_stream.data))
