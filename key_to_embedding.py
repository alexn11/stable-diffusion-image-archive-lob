import struct

import numpy as np

from BitStream import BitStream
from key_strings import convert_key_to_bit_stream

def convert_15_bits_int_to_float16_representation(datum_15_bits: int) -> int:
    #def convert_exponent_to_5_bits(datum: int):
    exp = (datum & 0b011110000000000) >> 10
    exp += 3
    sign = (datum & 0b100000000000000) >> 14
    datum = (datum & 0b000001111111111) | (exp << 10) | (sign << 15)
    return datum

def convert_float_to_15_bits(float_value: int):
    exp = (float_value & 0b0111110000000000) >> 10
    exp -= 3
    exp &= 0b1111
    sign = ((float_value & 0b1000000000000000) >> 15) & 1
    datum = (float_value & 0b000001111111111) | (exp << 10) | (sign << 14)
    #print(f'fv={float_value:016b}')
    #print(f's={sign:04b} - exp={exp:05b}')
    #print(f'cv={datum:016b}')
    return datum


def unpack_num_inference_steps(data_stream: BitStream) -> int:
    data_stream.set_chunk_size(2)
    num_inference_steps_level = data_stream.get_chunk()
    num_inference_steps = [ 12, 25, 36, 50 ][num_inference_steps_level]
    return num_inference_steps


def unpack_array(data_stream: BitStream,
                 array_type='prompt',
                 size=None,
                 debug=False) -> np.ndarray:
    chunk_size = 15 if(array_type == 'prompt') else 14
    if(size is None):
        size = 77*768-2 if(array_type == 'prompt') else 4*52*80
    data_stream.set_chunk_size(chunk_size)
    prompt_embeddings_data = data_stream.get_chunks(size)
    unpacked_bytes = bytearray(size * [0])
    for value_i, packed_value in enumerate(prompt_embeddings_data):
        unpacked_value = convert_15_bits_int_to_float16_representation(packed_value)
        if(debug):
            print(f'unpacked_value={unpacked_value:016b}')
        unpacked_bytes[2 * value_i + 1] = (unpacked_value >> 8) & 0xff
        unpacked_bytes[2 * value_i] = unpacked_value & 0xff
    prompt_embeddings_floats = struct.unpack(f'<{size}e', bytes(unpacked_bytes))
    # insert special values
    prompt_embeddings = prompt_embeddings_floats[:19]
    prompt_embeddings += [ -28.078125, ]
    prompt_embeddings += prompt_embeddings_floats[19:680]
    prompt_embeddings += [ 33.09375, ]
    prompt_embeddings += prompt_embeddings_floats[680:]
    return np.array(prompt_embeddings)

    

def unpack_key(base_64_key: str,
               debug=False, # i would use a logger if setting up the mode you want wasnt so complicated
               ) -> tuple: # int, tensor, tensor
    data_stream = convert_key_to_bit_stream(base_64_key,
                                            start_chunk_size_bits=2,
                                            data_size_bits=(77*768-2)*15+52*80*14+2)
    num_inference_steps = unpack_num_inference_steps(data_stream,)
    prompt_embeddings = unpack_prompt_embeddings(data_stream,
                                                 size=77*768-2,
                                                 debug=debug)
    #
    #print('unpacked bytes:')
    #for i in range(data_size):
    #    print(f'{unpacked_bytes[i]:08b}')
    #
    for value_i in range(data_size // 2):
        datum = (unpacked_bytes[2 * value_i + 1] << 8) | unpacked_bytes[2 * value_i]
        datum = convert_exponent_to_5_bits(datum)
        #print(f'datum={datum:016b}')
        unpacked_bytes[2 * value_i + 1] = (datum >> 8) & 0xff
        unpacked_bytes[2 * value_i] = datum & 0xff
    #
    return unpacked_bytes

def pack_float_array_into_binary_key(float16_array: np.ndarray, dbg=False) -> bytes:
    array_len = len(float16_array)
    #array_size_bytes = array_len * 2
    array_data = struct.unpack(f'{array_len}h', bytes(float16_array.data))
    packed_data_size_bits = array_len * 15
    packed_data_size_bytes = packed_data_size_bits // 8
    packed_data_nb_extra_bits = packed_data_size_bits % 8
    if(packed_data_nb_extra_bits > 0):
        packed_data_size_bytes += 1
    packed_data = bytearray(packed_data_size_bytes * [0])
    data_byte_i = 0
    datum_bit_i = 0
    current_datum = 0
    #print(f'current={current_datum:08b} - bit_i={datum_bit_i}')
    for array_value in array_data:
        #print(f'arrayv={array_value:016b}')
        packed_binary_value = convert_float_to_15_bits(array_value)
        packed_binary_value = (packed_binary_value & 0x7fff)
        #print(f'adding={packed_binary_value:015b}')
        # 1st bit
        shift_len = datum_bit_i
        mask_len = 8 - datum_bit_i
        mask = (1 << mask_len) - 1
        current_datum |= (packed_binary_value & mask) << shift_len
        packed_binary_value = (packed_binary_value >> mask_len)
        #print(f'current(0)={current_datum:08b}')
        packed_data[data_byte_i] = current_datum
        data_byte_i += 1
        # 2nd bit
        current_datum = packed_binary_value & 0xff
        #print(f'current(1)={current_datum:08b}')
        if(datum_bit_i != 0):
            packed_data[data_byte_i] = current_datum
            data_byte_i += 1
            packed_binary_value = (packed_binary_value >> 8) & 0xff
            current_datum = 0
            #print(f'current(2)={current_datum:08b}')
        else:
            current_datum &= 0x7f
            #print(f'current(1+)={current_datum:08b}')
        # 3rd bit
        if(datum_bit_i not in [0, 1]):
            mask_len = datum_bit_i - 1
            mask = (1 << mask_len) - 1
            current_datum = packed_binary_value & mask
            #print(f'current(2+)={current_datum:08b}')
        datum_bit_i = (datum_bit_i - 1) % 8
        #print(f'bit_i={datum_bit_i}')
    if(datum_bit_i > 0):
        #print(f'last extra={current_datum:08b}')
        packed_data[data_byte_i] = current_datum
        data_byte_i += 1
    return packed_data


def convert_bin_key_to_float_array(data: bytes, endian='<', dbg=False) -> bytes:
    data_size = len(data)
    nb_floats = data_size // 2
    float_array = struct.unpack(f'{endian}{nb_floats}e', data)
    float_array = np.array(float_array, dtype=np.float16)
    return float_array

def compute_embedding_and_latents_from_key(key: str|None = None,
                                           key_file_path: str|None = None,
                                           prompt_embeddings_size=77*768,
                                           latents_size=4*52*80):
    if(key is None):
        with open(key_file_path, 'r') as key_file:
            key = key_file.read().strip()
    key_bin = convert_key_to_binary(key, nb_bits_target=(prompt_embeddings_size + latents_size) * 15)
    prompt_embeddings_bin_size = 2 * prompt_embeddings_size
    latents_bin_size = 2 * latents_size
    data_bin_size = prompt_embeddings_bin_size + latents_bin_size
    #print(f'key bin len={len(key_bin)} - data size={data_bin_size} ({prompt_embeddings_bin_size}+{latents_bin_size})')
    data =  unpack_binary_key_into_binary_float_array(key_bin, data_size=data_bin_size)
    prompt_embeddings = convert_bin_key_to_float_array(data[:prompt_embeddings_bin_size],)
    # specific values
    prompt_embeddings[19] = -28.078125
    prompt_embeddings[681] = 33.09375
    #
    assert(len(prompt_embeddings) == prompt_embeddings_size)
    latents = convert_bin_key_to_float_array(data[prompt_embeddings_bin_size:])
    assert(len(latents)==latents_size)
    return prompt_embeddings, latents
