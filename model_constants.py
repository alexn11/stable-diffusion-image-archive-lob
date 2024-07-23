from functools import reduce

prompt_embeddings_shape = (77, 768)
prompt_embeddings_nb_special_values = 2
prompt_embeddings_nb_values = reduce(lambda x,y:x*y, prompt_embeddings_shape) - prompt_embeddings_nb_special_values
prompt_embeddings_bits_per_value = 15
prompt_embeddings_nb_bits = prompt_embeddings_nb_values * prompt_embeddings_bits_per_value
prompt_embeddings_nb_bytes = prompt_embeddings_nb_values * 2

latents_shape = (1, 4, 52, 80)
latents_nb_values = reduce(lambda x,y:x*y, latents_shape)
latents_bits_per_value = 14
latents_nb_bits = latents_nb_values * latents_bits_per_value
latents_nb_bytes = latents_nb_values * 2

num_inference_steps_nb_bits = 2
num_inference_steps_level_to_counts = [ 12, 25, 36, 50 ]

data_nb_bits = num_inference_steps_nb_bits + prompt_embeddings_nb_bits + latents_nb_bits

nb_padding_chars = 2