import base64
import random

from BitStream import BitStream
from model_constants import data_nb_bits
from model_constants import nb_padding_chars

base64_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

def generate_random_key_base64(nb_bits: int = data_nb_bits, nb_padding_chars=nb_padding_chars) -> str:
    if((nb_bits % 6) != 0):
        raise ValueError(f'expected multiple of 6 nb bits')
    length = nb_bits // 6
    base64_key = ''.join([ random.choice(base64_characters) for i in range(length) ])
    if(nb_padding_chars is not None):
        if(((nb_bits + 6*nb_padding_chars) % 8) != 0):
            raise ValueError(f'nb bits + 6*nb padding chars should be a multiple of 8')
        base64_key += 'A' * nb_padding_chars
    return base64_key

def convert_key_to_bit_stream(base_64_key: str,
                              start_chunk_size_bits=15,
                              data_size_bits=data_nb_bits,
                              nb_padding_chars=nb_padding_chars) -> BitStream:
    base_64_key += nb_padding_chars * 'A'
    key_bytes = base64.b64decode(base_64_key.encode('utf-8'))
    key_bit_stream = BitStream(key_bytes,
                               chunk_size_bits=start_chunk_size_bits,
                               data_size_bits=data_size_bits)
    return key_bit_stream