import struct

import numpy as np

from BitStream import BitStream
from model_constants import data_nb_bits
from model_constants import nb_padding_chars
from model_constants import num_inference_steps_nb_bits
from model_constants import num_inference_steps_level_to_counts, num_inference_steps_counts_to_level
from model_constants import prompt_embeddings_bits_per_value, latents_bits_per_value
from model_constants import prompt_embeddings_nb_bytes, latents_nb_bytes
from model_constants import prompt_embeddings_nb_values, latents_nb_values
from model_constants import prompt_emebeddings_special_values
from key_strings import convert_key_to_bit_stream, convert_packed_data_to_key

def convert_15_bits_int_to_float16_representation(datum_15_bits: int) -> int:
    #def convert_exponent_to_5_bits(datum: int):
    exp = (datum_15_bits & 0b011110000000000) >> 10
    exp += 3
    sign = (datum_15_bits & 0b100000000000000) >> 14
    datum = (datum_15_bits & 0b000001111111111) | (exp << 10) | (sign << 15)
    return datum

def convert_14_bits_int_to_float16_representation(datum_14_bits: int) -> int:
    exp = (datum_14_bits & 0b01110000000000) >> 10
    exp += 1
    sign = (datum_14_bits & 0b100000000000000) >> 13
    datum = (datum_14_bits & 0b000001111111111) | (exp << 10) | (sign << 15)
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

def convert_float_to_14_bits(float_value: int):
    exp = (float_value & 0b0111110000000000) >> 10
    exp -= 1
    exp &= 0b111
    sign = ((float_value & 0b1000000000000000) >> 15) & 1
    datum = (float_value & 0b000001111111111) | (exp << 10) | (sign << 13)
    #print(f'fv={float_value:016b}')
    #print(f's={sign:04b} - exp={exp:05b}')
    #print(f'cv={datum:016b}')
    return datum

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
    chunk_size = prompt_embeddings_bits_per_value if(array_type == 'prompt') else latents_bits_per_value
    if(chunk_size == 15):
        convert_function = convert_15_bits_int_to_float16_representation
    elif(chunk_size == 14):
        convert_function = convert_14_bits_int_to_float16_representation
    else:
        raise ValueError(f'unsupported chunk bit size: {chunk_size}')
    if(size_nb_values is None):
        size_nb_values = prompt_embeddings_nb_values if(array_type == 'prompt') else latents_nb_values
    if(size_bytes is None):
        size_bytes = prompt_embeddings_nb_bytes if(array_type == 'prompt') else latents_nb_bytes
    data_stream.set_chunk_size(chunk_size)
    packed_data = data_stream.get_chunks(size_nb_values)
    unpacked_bytes = bytearray(size_bytes * [0])
    for value_i, packed_value in enumerate(packed_data):
        unpacked_value = convert_function(packed_value)
        if(debug):
            print(f'unpacked_value={unpacked_value:016b}')
        unpacked_bytes[2 * value_i + 1] = (unpacked_value >> 8) & 0xff
        unpacked_bytes[2 * value_i] = unpacked_value & 0xff
    array_as_list = list(struct.unpack(f'<{size_nb_values}e', bytes(unpacked_bytes)))
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
    return np.array(array_as_list)

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
               ) -> tuple: # int, tensor, tensor
    data_stream = convert_key_to_bit_stream(base_64_key,
                                            start_chunk_size_bits=2,
                                            data_size_bits=data_nb_bits,
                                            nb_padding_chars=nb_padding_chars)
    num_inference_steps = unpack_num_inference_steps(data_stream,)
    prompt_embeddings = unpack_prompt_embeddings(data_stream,
                                                 debug=debug)
    latents = unpack_latents(data_stream, debug=debug)
    return num_inference_steps, prompt_embeddings, latents

def pack_num_inference_steps(packed_data_stream: BitStream, num_inference_steps: int) -> BitStream:
    packed_data_stream.set_chunk_size(num_inference_steps_nb_bits)
    packed_data_stream.write_chunk(num_inference_steps_counts_to_level[num_inference_steps])
    return packed_data_stream

def pack_array(packed_data_stream: BitStream, array: np.ndarray, array_type='', debug = False) -> BitStream:
    assert(array_type in ['prompt', 'latents'])
    chunk_size_bits = prompt_embeddings_bits_per_value if(array_type == 'prompt') else latents_bits_per_value
    if(chunk_size_bits == 15):
        convert_function = convert_float_to_15_bits
    elif(chunk_size_bits == 14):
        convert_float_to_14_bits
    else:
        raise ValueError(f'unsupported chunk bit size: {chunk_size_bits}')
    packed_data_stream.set_chunk_size(chunk_size_bits)
    if(array_type == 'prompt'):
        # remove special values
        array = np.delete(array, list(prompt_emebeddings_special_values.keys()))
    array_len = len(array)
    if(array_type == 'prompt'):
        assert(array_len == prompt_embeddings_nb_values)
    else:
        assert(array_len == latents_nb_values)
    array_data = struct.unpack(f'{array_len}h', bytes(array.data))
    for value in array_data:
        packed_value = convert_function(value)
        packed_data_stream.write_chunk(packed_value)
    return packed_data_stream

def pack_prompt_embeddings(packed_data_stream: BitStream, prompt_embeddings: np.ndarray, debug = False) -> BitStream:
    # remove special values
    return pack_array(packed_data_stream, prompt_embeddings, array_type='prompt', debug=debug)

def pack_prompt_latents(packed_data_stream: BitStream, latents: np.ndarray, debug = False) -> BitStream:
    return pack_array(packed_data_stream, latents, array_type='latents', debug=debug)

def pack_data_into_key(num_inference_steps: int,
                       prompt_embeddings: np.ndarray,
                       latents: np.ndarray,
                       debug = False,
                       return_type='key') -> str:
    packed_data_stream = BitStream(data_size_bits=data_nb_bits, mode='w')
    pack_num_inference_steps(packed_data_stream, num_inference_steps)
    pack_prompt_embeddings(packed_data_stream, prompt_embeddings, debug=debug)
    pack_prompt_latents(packed_data_stream, latents, debug=debug)
    if(return_type != 'key'):
        return packed_data_stream
    return convert_packed_data_to_key(bytes(packed_data_stream.data))
