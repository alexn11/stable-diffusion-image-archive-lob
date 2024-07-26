from functools import reduce

prompt_embeddings_shape = (77, 768)
prompt_embeddings_nb_special_values = 2
prompt_embeddings_nb_values = reduce(lambda x,y:x*y, prompt_embeddings_shape) - prompt_embeddings_nb_special_values
prompt_embeddings_bits_per_value = 15
prompt_embeddings_exponent_max = 3
prompt_embeddings_nb_bits = prompt_embeddings_nb_values * prompt_embeddings_bits_per_value
prompt_embeddings_nb_bytes = prompt_embeddings_nb_values * 2
prompt_embeddings_special_values = {
    19: -28.078125,
    681: 33.09375,
}
#prompt_embeddings_special_values = {
#    19: 12.086,
#    681: 17.1,
#}
#prompt_embeddings_special_values = {
#    19: -15.9921875,
#    681: 15.9921875,
#}

latents_shape = (1, 4, 52, 80)
latents_nb_values = reduce(lambda x,y:x*y, latents_shape)
latents_bits_per_value = 15
latents_exponent_max = 0
latents_nb_bits = latents_nb_values * latents_bits_per_value
latents_nb_bytes = latents_nb_values * 2

#num_inference_steps_nb_bits = 2
#num_inference_steps_level_to_counts = [ 12, 25, 36, 50 ]
#num_inference_steps_nb_bits = 0
#num_inference_steps_level_to_counts = [ ]
num_inference_steps_nb_bits = 6
num_inference_steps_level_to_counts = [ i+1 for i in range(64) ]
num_inference_steps_counts_to_level = {
    counts: level
    for level, counts in enumerate(num_inference_steps_level_to_counts)
}

data_nb_bits = num_inference_steps_nb_bits + prompt_embeddings_nb_bits + latents_nb_bits

#nb_padding_chars = 2
nb_padding_chars = 0

key_length = data_nb_bits // 6

image_height = 416
image_width = 640


