from key_strings import generate_random_key_base64

key_size_bits = (77*768-2)*15 + 52*80*14 + 2

generated_key = generate_random_key_base64(key_size_bits, nb_padding_chars=2)
with open('test-key.txt', 'w') as key_file:
    key_file.write(generated_key)