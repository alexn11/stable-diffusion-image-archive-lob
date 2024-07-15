import base64
import random
import struct

import numpy as np

base64_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

def generate_random_base64(nb_bytes: int = 118268) -> str:
    # 110880 == 77 * 768 * 15 / 8
    length = nb_bytes * 8 // 6
    return ''.join([ random.choice(base64_characters) for i in range(length) ])

def convert_key_to_binary(key: str) -> bytes:
    return base64.b64decode(key + '==')

def convert_exponent_to_5_bits(datum: int):
    exp = (datum & 0b011110000000000) >> 10
    exp += 3
    sign = (datum & 0b100000000000000) >> 14
    datum = (datum & 0b000001111111111) | (exp << 10) | (sign << 15)
    return datum

def unpack_binary_key_into_binary_float_array(key_bin: bytes, data_size=(77*768)*2) -> bytes:
    data = bytearray(data_size * [0])
    datum = 0
    datum_bit_i = 0
    data_i = 0
    for key_byte_bit_i in range(0, 8*len(key_bin), 15):
        key_byte_i = key_byte_bit_i // 8
        datum = 0
        for datum_bit_i in range(15):
            key_bit_i = key_byte_bit_i + datum_bit_i
            if(key_bit_i < len(key_bin)*8):
                datum = (datum << 1) | ((key_bin[key_byte_i] >> (7 - key_bit_i % 8)) & 1)
        datum = convert_exponent_to_5_bits(datum)
        if(data_i >= data_size):
            break
        data[data_i] = (datum >> 8) # % 256
        data[data_i + 1] = datum % 256
        data_i += 2
        # special fixed values
        if(data_i == 19*2):
            data_i += 2
        elif(data_i == 681*2):
            data_i += 2
    return bytes(data)

def convert_bin_key_to_float_array(key: bytes, data_size=(77*768)*2) -> bytes:
    data =  unpack_binary_key_into_binary_float_array(key)
    float_array = struct.unpack(f'>{data_size//2}e', data)
    # special fixed values
    float_array = np.array(float_array)
    float_array[19] = -28.078125
    float_array[681] = 33.09375
    return float_array

def compute_embedding_from_key(key: str|None = None,
                               key_file_path: str|None = None,
                               ):
    if(key is None):
        with open(key_file_path, 'r') as key_file:
            key = key_file.read().strip()
    key_bin = convert_key_to_binary(key)
    return convert_bin_key_to_float_array(key_bin)