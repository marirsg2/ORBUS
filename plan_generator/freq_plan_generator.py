"""
:summary: given a distribution of size=1 features, generates plans according to that distribution and pickles it.
"""

import pickle
import random
NUM_PLANS_NEEDED = 10000
NUM_FEATURES = 40
HIGH_OCCURRENCE_COUNT =300 #these counts are used as weights for generating plans
MED_OCCURRENCE_COUNT =100
LOW_OCCURRENCE_COUNT =10
MAX_LENGTH = 4 #prob that the plan will end with this step
RATIO_HIGH_FREQ_FEATURES = 0.03
RATIO_MED_FREQ_FEATURES = 0.2

dest_pickle_file_name = "default_plans_pool.p"
all_plans = set()
all_features = set()
# s1_features = [str(x) for x in "abcdefghijklmnopqrstuvwxyz"] #are represented by letters
s1_features = [str(x) for x in range(1,NUM_FEATURES+1)] #are represented by letters
s1_features = s1_features[:NUM_FEATURES]
s1_counts= [LOW_OCCURRENCE_COUNT]*len(s1_features) # this will EITHER be converted into probabilities.or be used as weights to sample
num_high_freq = int(RATIO_HIGH_FREQ_FEATURES*NUM_FEATURES)
num_med_freq = int(RATIO_MED_FREQ_FEATURES*NUM_FEATURES) #the rest are low freq
chosen_features = set()

high_freq_s1_features = random.sample(set(s1_features).difference(chosen_features), num_high_freq)
chosen_features.update(high_freq_s1_features)
for single_feat in high_freq_s1_features:
    s1_counts[s1_features.index(single_feat)] = HIGH_OCCURRENCE_COUNT
#---end for
med_freq_features = random.sample(set(s1_features).difference(chosen_features),num_med_freq)
chosen_features.update(med_freq_features)
for single_feat in med_freq_features:
    s1_counts[s1_features.index(single_feat)] = MED_OCCURRENCE_COUNT
#--end for
#---now the counts will serve as weights for sampling

#todo add check to see if the num of all possible permutations can be generated from the num plans
while len(all_plans) < NUM_PLANS_NEEDED:
    plan_length = random.randint(1,MAX_LENGTH) #includes max length
    curr_plan = random.choices(s1_features,weights = s1_counts,k=plan_length)
    curr_plan = [(x,) for x in curr_plan ]
    # curr_plan = {x:1 for x in curr_plan}
    all_plans.add(tuple(curr_plan))
#end while
all_plans_formatted = []
for single_plan in all_plans:
    mod_single_plan = set()
    for step in single_plan:
        mod_single_plan = mod_single_plan.union({x for x in step })
    #end for loop
    all_plans_formatted.append(mod_single_plan)
#end outer for loop
all_plans = all_plans_formatted

with open(dest_pickle_file_name, "wb") as dest:
    pickle.dump(all_plans,dest)
print(all_plans)
print(len(all_plans))

# with open("OLD_default_plans_pool.p" ,"rb") as src:
#     a = pickle.load(src)
#     print(a)