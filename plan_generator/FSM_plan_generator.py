"""
:summary: given a distribution of size=1 features, generates plans according to that distribution and pickles it.
"""

import pickle
import random
import numpy as np
NUM_PLANS_NEEDED = 10000
DEFAULT_OCCURRENCE_COUNT =10
MAX_LENGTH = 6 #prob that the plan will end with this step
edited_feature_counts_sparse_dict = \
{
    "beach" : [("burger",300),("movie",300)],
    "burger" : [("movie",300),("coffee",100)],
    "zoo" : [("veg",500)],
    "art_museum" : [("library",100),("movie",300)],
    "coffee" : [("pastry",500),("shopping",100)],
}


dest_pickle_file_name = "default_plans_pool.p"
all_plans = set()
list_letters = [x for x in "abcdefghijklmnopqrstuvwxyz"] #how the features are finally encoded into plan
list_features = ["zoo", "beach", "burger", "coffee", "movie", "pastry", "shopping", "veg", "library", "meat", "spicy", "art_museum"] # the actual features
num_features = len(list_features)
FSM_counts = np.array([[DEFAULT_OCCURRENCE_COUNT] * num_features] * num_features)
np.fill_diagonal(FSM_counts,0)
#update the FSM counts with the input dict
for single_feat_key in edited_feature_counts_sparse_dict.keys():
    for single_feat_count_pair in edited_feature_counts_sparse_dict[single_feat_key]:
        dest_feat = single_feat_count_pair[0]
        dest_count = single_feat_count_pair[1]
        FSM_counts[list_features.index(single_feat_key)][list_features.index(dest_feat)] = dest_count
#end outer for loop
#normalize the FSM counts to get probabilities
FSM_freq = FSM_counts/np.sum(FSM_counts,axis=0)
print(FSM_freq)
#generate plans based on those probabilities
#todo add check to see if the num of all possible permutations can be generated from the num plans
while len(all_plans) < NUM_PLANS_NEEDED:
    plan_length = random.randint(1,MAX_LENGTH) #includes max length
    curr_plan = [(random.choice(list_features),)]
    for _ in range(plan_length):
        prior_feature = curr_plan[-1][0]
        next_feat_weights = FSM_freq[:, list_features.index(prior_feature)]
        next_feature = random.choices(list_features, weights = next_feat_weights, k=1)[0]
        curr_plan.append((next_feature,))
    all_plans.add(tuple(curr_plan))
#end while
all_plans_formatted = []
for single_plan in all_plans:
    mod_single_plan = []
    for step in single_plan:
        mod_single_plan.append({list_letters[list_features.index(x)] :1 for x in step})
    #end for loop
    all_plans_formatted.append(mod_single_plan)
#end outer for loop
all_plans = all_plans_formatted

with open(dest_pickle_file_name, "wb") as dest:
    pickle.dump(all_plans,dest)
print(all_plans)
print(len(all_plans))

