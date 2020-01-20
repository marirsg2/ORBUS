"""
:summary: given a distribution of size=1 features, generates plans according to that distribution and pickles it.
"""

import pickle
import random
import datetime
import numpy as np

#--------------------------------
NUM_PLANS_NEEDED = 10000
NUM_FEATURES = 400
NUM_GROUPS = 1
NUM_FEATURES = int(NUM_FEATURES/NUM_GROUPS)
distribution_samples = np.random.normal(150,50,NUM_FEATURES)
MIN_FEATURES = 4 #prob that the plan will end with this step
MAX_FEATURES = 8 #prob that the plan will end with this step

# HIGH_OCCURRENCE_COUNT = 1350 #these counts are used as weights for generating plans
# MED_OCCURRENCE_COUNT = 450
# LOW_OCCURRENCE_COUNT = 150

# RATIO_HIGH_FREQ_FEATURES = 0.06 #this is the NUMBER of features, what constitutes as high occurrence (probability) is given by the COUNTS defined previously
# RATIO_MED_FREQ_FEATURES = 0.88 #THE REST ARE LOW FREQ

# date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
# date_time_str = date_time_str.replace(" ", "_")
# date_time_str = date_time_str.replace("/", "_")
# date_time_str = date_time_str.replace(",", "_")
# date_time_str = date_time_str.replace(":", "_")
# print("date and time:", date_time_str)
# dest_pickle_file_name = "plans_pool" + date_time_str +".p"

dest_pickle_file_name = "default_plans_pool.p"

all_plans = set()
all_features = set()
# s1_features = [str(x) for x in "abcdefghijklmnopqrstuvwxyz"] #are represented by letters
s1_features = [str(x) for x in range(1,NUM_FEATURES+1)] #are represented by letters
s1_features = s1_features[:NUM_FEATURES]
s1_weights = distribution_samples

#
# #todo add check to see if the num of all possible permutations can be generated from the num plans
while len(all_plans) < NUM_PLANS_NEEDED:
    plan_feature_num = random.randint(MIN_FEATURES, MAX_FEATURES) #includes max length
    curr_plan = set()
    plan_group_num = random.randint(1, NUM_GROUPS)
    if plan_group_num == 2:
        plan_feature_num *=2 # we double the number of features
    while len(curr_plan) < plan_feature_num:
        choice = random.choices(s1_features)#
        curr_plan.add("g"+str(plan_group_num)+"_"+ choice[0])
    # curr_plan = {x:1 for x in curr_plan}
    all_plans.add(tuple(curr_plan))
#end while


# while len(all_plans) < NUM_PLANS_NEEDED:
#     plan_feature_num = random.randint(MIN_FEATURES, MAX_FEATURES) #includes max length
#     curr_plan = set()
#     plan_group_num = random.randint(1, NUM_GROUPS)
#     while len(curr_plan) < plan_feature_num:
#         choice = random.choices(s1_features, weights = s1_weights, k=1)
#         curr_plan.add("g"+str(plan_group_num)+"_"+ choice[0])
#     # curr_plan = {x:1 for x in curr_plan}
#     all_plans.add(tuple(curr_plan))
# #end while

temp_all_plans = [set(x) for x in all_plans]
all_plans = temp_all_plans
with open(dest_pickle_file_name, "wb") as dest:
    pickle.dump(all_plans,dest)
print(all_plans)
print(len(all_plans))
