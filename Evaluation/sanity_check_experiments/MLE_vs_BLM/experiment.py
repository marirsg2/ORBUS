from Evaluation.sanity_check_experiments.MLE_vs_BLM.line_weights import oracle
import pickle
import numpy as np
from pprint import pprint
from itertools import permutations

class experiment:
    def __init__(self):
        self.s1_features = ["zoo", "beach", "burger", "coffee", "movie", "pastry", "shopping", "veg", "library", "meat", "spicy_food", "art_museum"]
        self.oracle = oracle(self.s1_features, l_n_max=1, probability_per_level=(0.75, ), gaussian_noise_sd=0)
        self.backlog_plans = pickle.load(open("default_plans_pool.p", "rb"))
        pass

    def test_oracle_feedback(self):
        rand_idx = np.random.randint(0, len(self.backlog_plans))
        rand_plan = list(self.backlog_plans)[rand_idx]
        print("Random Plan : ")
        pprint(rand_plan)
        plan_features = []
        for feature_level in range(1, self.oracle.max_level + 1):
            for single_feature in permutations(rand_plan, feature_level):
                plan_features.append(single_feature)
        pprint(plan_features)
        pprint("Oracle's relevant features:")
        pprint(self.oracle.feature_preferences_dict)
        pprint(self.oracle.get_feedback([[rand_plan, plan_features]]))

if __name__ == "__main__":
    exp = experiment()
