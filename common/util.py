from itertools import product, combinations
import re

def get_all_possible_features_till_level(l1_features, l_max):
    feature_level_list = [l + 1 for l in range(l_max)]
    all_features = []
    for current_feature_level in feature_level_list:
        features_at_current_level = product(l1_features, repeat=current_feature_level)
        for single_feature in features_at_current_level:
            formatted_feature = "f-"
            for single_variable in single_feature:
                formatted_feature += re.split('f-', single_variable)[-1]
            all_features.append(formatted_feature)
    return all_features

def get_all_features_in_plan(plan, l_max):
    """

    :param plan: state sequence with x,y,l-1-feature as keys
    :param l_max:
    :return:
    """
    l1_features = []
    for single_state in plan:
        curr_vars = set(single_state.keys())
        curr_vars.remove("x")
        curr_vars.remove("y")

        for single_feature in curr_vars:
            l1_features.append(single_feature)

    feature_level_list = [l + 1 for l in range(l_max)]
    all_features = set()
    for current_feature_level in feature_level_list:
        features_at_current_level = combinations(l1_features, current_feature_level) # this will retain the sequence order
        for single_feature in features_at_current_level:
            formatted_feature = ""
            for single_variable in single_feature:
                formatted_feature += re.split('f-', single_variable)[-1]
            all_features.add(formatted_feature)
    return all_features