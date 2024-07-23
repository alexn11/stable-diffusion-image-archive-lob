import struct

import numpy as np

from BitStream import BitStream
from model_constants import data_nb_bits
from model_constants import nb_padding_chars
from model_constants import num_inference_steps_nb_bits, num_inference_steps_level_to_counts
from model_constants import prompt_embeddings_bits_per_value, latents_bits_per_value
from model_constants import prompt_embeddings_nb_bytes, latents_nb_bytes
from model_constants import prompt_embeddings_nb_values, latents_nb_values
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
    if(size_nb_values is None):
        size_nb_values = prompt_embeddings_nb_values if(array_type == 'prompt') else latents_nb_values
    if(size_bytes is None):
        size_bytes = prompt_embeddings_nb_bytes if(array_type == 'prompt') else latents_nb_bytes
    data_stream.set_chunk_size(chunk_size)
    packed_data = data_stream.get_chunks(size_nb_values)
    unpacked_bytes = bytearray(size_bytes * [0])
    for value_i, packed_value in enumerate(packed_data):
        if(array_type == 'prompt'):
            unpacked_value = convert_15_bits_int_to_float16_representation(packed_value)
        else:
            unpacked_value = convert_14_bits_int_to_float16_representation(packed_value)
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

# here: TODO - write
def pack_data_into_key(num_inference_steps: int,
                       prompt_embeddings: np.ndarray,
                       latents: np.ndarray,
                       debug = False,
                       return_type='key') -> str:
    # TODO from here (until "below is DONE")
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
    # below is DONE:
    if(return_type != 'jey'):
        return packed_data
    return convert_packed_data_to_key(packed_data)
