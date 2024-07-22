from key_to_embedding import generate_random_key_base64

nb_shorts = (77*768+4*52*80)
generated_key = generate_random_key_base64(nb_shorts)
with open('test-key.txt', 'w') as key_file:
    key_file.write(generated_key)