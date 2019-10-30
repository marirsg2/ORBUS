"""

"""
from enum import IntEnum
import tester.testing_manager as tm
import sys


class AL_variants(IntEnum):
    RANDOM_W_DIVERSITY = 0
    VARIANCE = 1
    VARIANCE_W_FEATURE_COUNT = 2
    RBUS = 3
    RBUS_W_FEATURE_COUNT = 4
    NUM_AL_VARIANTS = 5 #SINCE THE FIRST ONE IS 0 BASED index
default_stdout = sys.stdout
starting_trial_num = 5 #0 indexed, can start from > 0
stopping_trial_num = 5 #
num_variants = int(AL_variants.NUM_AL_VARIANTS) #this is just to remind you to test (1)Random(2)
total_experiments = num_variants*stopping_trial_num
starting_experiment_num = num_variants*starting_trial_num
stopping_experiment_num = num_variants*stopping_trial_num
noise_in_ratings = 0.9
results_file_prefix = "scale_test_noise"+str(noise_in_ratings)+"_"
for i in range(starting_experiment_num,stopping_experiment_num+1,num_variants): #+1 since the last index is not used
    manager_pickle_file = "manager_ver_" + str(int(i/num_variants)+1) +".p"
    for j in range(num_variants):
        experiment_idx = i+j
        sys.stdout = default_stdout
        print("Running Trial ",experiment_idx, " out of " , total_experiments)
        sys.stdout = open(results_file_prefix+str(experiment_idx)+".result",'w')
        testing_args_order = ("total_num_plans" , "plans_per_round","random_seed","noise_value","random_sampling_enabled",\
                                    "include_gain","include_feature_distinguishing","manager_pickle_file" )
        testing_args_dict = {}
        if j == AL_variants.RANDOM_W_DIVERSITY:
            testing_args_dict = {"total_num_plans" : 120, "plans_per_round" : 30, "random_seed" : 150, "noise_value" : noise_in_ratings, "random_sampling_enabled" : True,\
                                    "include_gain" : False,"include_feature_distinguishing":False,"manager_pickle_file" : manager_pickle_file}
        elif j == AL_variants.VARIANCE:
            testing_args_dict = {"total_num_plans" : 120, "plans_per_round" : 30, "random_seed" : 150, "noise_value" : noise_in_ratings, "random_sampling_enabled" : False,\
                                    "include_gain" : False,"include_feature_distinguishing":False,"manager_pickle_file" : manager_pickle_file}
        elif j == AL_variants.VARIANCE_W_FEATURE_COUNT:
            testing_args_dict = {"total_num_plans" : 120, "plans_per_round" : 30, "random_seed" : 150, "noise_value" : noise_in_ratings, "random_sampling_enabled" : False,\
                                    "include_gain" : False,"include_feature_distinguishing":True,"manager_pickle_file" : manager_pickle_file}
        elif j == AL_variants.RBUS:
            testing_args_dict = {"total_num_plans" : 120, "plans_per_round" : 30, "random_seed" : 150, "noise_value" : noise_in_ratings, "random_sampling_enabled" : False,\
                                    "include_gain" : True,"include_feature_distinguishing":False,"manager_pickle_file" : manager_pickle_file}
        elif j == AL_variants.RBUS_W_FEATURE_COUNT:
            testing_args_dict = {"total_num_plans" : 120, "plans_per_round" : 30, "random_seed" : 150, "noise_value" : noise_in_ratings, "random_sampling_enabled" : False,\
                                    "include_gain" : True,"include_feature_distinguishing":True,"manager_pickle_file" : manager_pickle_file}

        testing_args_list = [testing_args_dict[x] for x in testing_args_order]
        tm.Active_Learning_Testing(*testing_args_list)
        print(" TESTED VARIANT ", str(j) )
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")



