import numpy as np
chains = 1
no_of_samples = 1500
number_of_dimensions = 12
sigma = 5 # setting a high sigma else we don't get uncertainty
sd_for_priors = 10
max_training_data = 20
step_size = 2
training_data_count_values = np.arange(number_of_dimensions+1,
                                         max_training_data + 1,
                                         step_size).astype(int)