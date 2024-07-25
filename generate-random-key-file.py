from model_constants import data_nb_bits
from key_strings import generate_random_key_base64

key_size_bits = data_nb_bits

generated_key = generate_random_key_base64(key_size_bits, nb_padding_chars=None)
with open('test-key.txt', 'w') as key_file:
    key_file.write(generated_key)