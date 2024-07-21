from key_to_embedding import generate_random_base64

data_size = (77*768+4*52*80)*2
generated_key = generate_random_base64(data_size)
with open('test-key.txt', 'w') as key_file:
    key_file.write(generated_key)