import numpy as np
import pickle
import math
from manager.main_manager import Manager
# from common.system_logger.general_logger import logger
from common.visualizations.general_viz import GenerateVisualizations
from learning_engine.bayesian_linear_model import bayesian_linear_model
from sklearn import linear_model
from linetimer import CodeTimer
from scipy.stats import describe as summ_stats_fnc
import matplotlib
from matplotlib import pyplot as plt
import random
matplotlib.use("TkAgg")
#====================================================


def test_full_cycle_and_accuracy(test_size, num_rounds, num_plans_per_round, random_sampling_enabled = False,
                                 include_gain = False, include_feature_distinguishing=True,include_prob_term = True,
                                 random_seed = 40,
                                 manager_pickle_file = None, input_rating_noise =0.0,
                                 prob_feat_select= 0.2, preference_distribution_string="power_law"):

    learn_LSfit = True
    print("doing probability per level =", prob_feat_select)
    print("RANOM SAMPLING ENABLED = ",random_sampling_enabled)
    print("Rating Noise =", input_rating_noise)
    in_region_error = []
    out_region_error = []
    bayes_error_list = []
    MLE_error_list = []
    inRegion_bayes_error_list = []
    inRegion_MLE_error_list = []
    annotated_test_plans = []
    try:
        with open(manager_pickle_file,"rb") as src:
            manager = pickle.load(src)
            annotated_test_plans = pickle.load(src)
            pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
            pref_list = sorted(pref_list, key=lambda x: x[0])
            print("TRUE FEATURE PREFERENCE DICT", pref_list)
            print("num features =", len(pref_list))
            print("-=-=-=-= USING PICKLED FILE-=-=-=-=-=-")
    except :
        RATING_NOISE = input_rating_noise

        manager = Manager(prob_feature_selection=prob_feat_select,
                          random_seed=random_seed, preference_distribution_string=preference_distribution_string)
        pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
        pref_list = sorted(pref_list, key=lambda x: x[0])
        print("TRUE FEATURE PREFERENCE DICT", pref_list)
        print("num features =", len(pref_list))
        print("Include gain is =", include_gain)
        manager.sim_human.change_rating_noise(0.0)# todo NOTE the test dataset has no noise.
        test_plans = manager.get_extremities_test_set_before_training(test_set_size=1000) #0.1 top and 0.1 bottom
        annotated_test_plans = manager.get_feedback(test_plans)
        manager.sim_human.change_rating_noise(input_rating_noise)  # todo NOTE the test dataset has no noise.
        manager.test_set = annotated_test_plans
        with open(manager_pickle_file, "wb")as dest:
            pickle.dump(manager,dest)
            pickle.dump(annotated_test_plans,dest)
    #end except for when the pickle file is not found
    print("finished initializing manager, starting process")
    scores = []
    for plan in annotated_test_plans:
        scores.append(plan[-1])
    print("TEST set scores stats")
    print(summ_stats_fnc(scores))
    print(sorted(scores))

    manager.sim_human.change_rating_noise(input_rating_noise)  # SET NOISE IN RATING

    random.seed(random_seed)

    for round_num in range(num_rounds):
        print("============ ROUND ",round_num,"====================")

        if random_sampling_enabled:
            sampled_plans = manager.sample_randomly(num_plans_per_round)
        else:
            sampled_plans = manager.get_plans_for_round(num_plans_per_round)

        annotated_plans = manager.get_feedback(sampled_plans)
        #analyze the plans sampled. For each plan print the number of s1 features, s2 features,etc WITH their associated
        # freq and score
        sampled_plan_analytics = []
        for single_plan in annotated_plans:
            #get the liked and disliked features
            all_features = single_plan[0]
            feature_analytics = []
            feature_count_by_size = {}
            for single_feature in all_features:
                try:
                    feature_count_by_size[len(single_feature)] += 1
                except:
                    feature_count_by_size[len(single_feature)] = 1
                try:
                    feature_analytics.append((single_feature,manager.freq_dict[single_feature]*abs(manager.sim_human.feature_preferences_dict[single_feature][1]),
                        manager.freq_dict[single_feature],manager.sim_human.feature_preferences_dict[single_feature]))
                except KeyError: #happens when some features are not in the freq dict, (very low prob)
                    pass

            #end for loop over all features
            feature_analytics = sorted(feature_analytics, key = lambda x: x[1],reverse=True)
            sampled_plan_analytics.append(( sum([x[1] for x in feature_count_by_size.items()]) , feature_count_by_size, feature_analytics))
        #--end for loop
        print(" SAMPLED PLANS ANALYTICS =", sampled_plan_analytics)
        if round_num == num_rounds-1:
            learn_LSfit = True
        #end if
        manager.update_indices(annotated_plans)

        manager.relearn_model(learn_LSfit, num_chains=2) # here is where we first train the model
        #todo REMOVE THIS and maybe move it to replace the test set
        # manager.select_best_and_worst(30)
        bayes_error, MLE_error = manager.evaluate(annotated_test_plans)
        bayes_error_list.append(bayes_error)
        MLE_error_list.append(MLE_error)
        inRegion_bayes_error, inRegion_MLE_error = manager.region_based_evaluation(annotated_test_plans, [(0.0, 0.1), (0.9, 1.0)],
                                        inside_region=True)  # the second parameter is percentile regions to evaluate in
        out_region_error.append(manager.region_based_evaluation(annotated_test_plans, [(0.0, 0.1), (0.9, 1.0)],
                                        inside_region=False) ) # the second parameter is percentile regions to evaluate in
        inRegion_bayes_error_list.append(inRegion_bayes_error)
        inRegion_MLE_error_list.append(inRegion_MLE_error)

    #end for loop through
    #---now measure the accuracy




    #TODO VERY IMPORTANT TO UPDATE INDICES AFTER THE EVALUATION, else all the weights are not correctly learned
    # todo analyse the learnt model to see if all the MCMC chains are agreeing with each other

    # currently we are testing with gain  = 1, and only variance.
    # ALSO increase noise and compare performance. The difference in region and not region is expected to be more.
    # todo analyse the learnt model to see if all the MCMC chains are agreeing with each other

    # manager.learning_model.plot_learnt_parameters()
    pref_dict = dict(pref_list)
    print("Include gain is =", include_gain)
    print("The preference probabilities are", manager.sim_human.probability_per_level)
    print("TRUE FEATURE PREFERENCE", pref_list)
    print("FREQ DICT", manager.compute_freq_dict(manager.formatted_plan_dataset,manager.relevant_features))
    print("num features =", len(pref_list), "gain included is ",include_gain)
    print("FEATURES DISCOVERED ", manager.relevant_features_dimension , " ", [(x,pref_dict[x]) for x in manager.relevant_features])
    print("IN REGION ERROR =" , in_region_error)
    print("OUT OF REGION ERROR =" ,out_region_error)
    print("============================================================")
    print("BAYES ERROR LIST ", bayes_error_list)
    print("MLE ERROR LIST ", MLE_error_list)
    print("INTERESTING REGION BAYES ERROR LIST ", inRegion_bayes_error_list)
    print("INTERESTING REGION  MLE ERROR LIST ", inRegion_MLE_error_list)
    print("MIN,MAX",manager.min_rating,manager.max_rating)

    #todo remove plotting code for speed
    #manager.learning_model_bayes.plot_learnt_parameters()
    print("RANOM SAMPLING ENABLED = ",random_sampling_enabled)
    print("Rating Noise =", input_rating_noise)
    print("manager file =", manager_pickle_file)
    print("used Indices (testplans+annot) =", manager.indices_used)

    return in_region_error,out_region_error, bayes_error_list,MLE_error_list
#====================================================




def test_bayesian_MV_linModel(toy_data_input,toy_data_output):
    """

    :param toy_data:
    :return:
    """
    toy_data_input = list(toy_data_input)
    toy_data_output = list(toy_data_output)
    toy_data = list(zip(toy_data_input,toy_data_output))
    num_points = len(toy_data_input)
    num_dimensions = toy_data_input[0].shape[0]
    prior_weights = np.random.rand(num_dimensions)
    model_b = bayesian_linear_model()
    model_b.learn_bayesian_linear_model(toy_data, prior_weights, num_dimensions, sd=1.0,sampling_count=800,num_chains=2)
    model_b.plot_learnt_parameters()
    return [np.mean(np.array(model_b.linear_params_values["betas"][0:2000,x])) for x in range(num_dimensions)],\
            np.mean(np.array(model_b.linear_params_values["alpha"][0:2000])),model_b.linear_params_values

def test_basic_MV_linModel(toy_data_input, toy_data_output):
    reg = linear_model.LinearRegression()
    reg.fit(toy_data_input, toy_data_output)
    print("Coefficients's values ", reg.coef_)
    print("Intercept: %.4f" % reg.intercept_)


def Active_Learning_Testing(total_num_plans = 240, plans_per_round = 30, random_seed = 150, noise_value = 1.0, random_sampling_enabled = False,
                            include_gain=True, include_feature_distinguishing=True, include_prob_term=True, manager_pickle_file = "default_man.p",
                            prob_feat_select= 0.2, preference_distribution_string="power_law"):


    print ("doing probability per level =", prob_feat_select)
    print(manager_pickle_file)
    for i in range(1):
        with CodeTimer():
            NOgain_in_region_error,NOgain_out_region_error, NOgain_bayes_error_list,NOgain_MLE_error_list = \
                test_full_cycle_and_accuracy(test_size=1000, num_rounds = int(total_num_plans/plans_per_round),
                                             num_plans_per_round = plans_per_round,
                                             random_sampling_enabled = random_sampling_enabled, include_gain=include_gain, include_feature_distinguishing=include_feature_distinguishing,
                                             include_prob_term = include_prob_term,
                                             random_seed= random_seed, manager_pickle_file=manager_pickle_file, input_rating_noise=noise_value,
                                             prob_feat_select=prob_feat_select, preference_distribution_string=preference_distribution_string)



# x_axis = [plans_per_round*x for x in range(1,int(total_num_plans/plans_per_round)+1)]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# y_axis = NOgain_MLE_error_list
# plt.plot(x_axis,y_axis, color = 'r')
# y_axis = NOgain_bayes_error_list
# plt.plot(x_axis, y_axis, color='b')
# #----
# ax2 = fig.add_subplot(2,1,2)
# y_axis = MLE_error_list
# plt.plot(x_axis, y_axis, color='r')
# y_axis = bayes_error_list
# plt.plot(x_axis,y_axis, color = 'b')
# plt.show()



if __name__ == "__main__":


    Active_Learning_Testing(total_num_plans = 210, plans_per_round = 30, random_seed = 150, noise_value = 0.6, random_sampling_enabled = True,
                            include_gain=False, include_feature_distinguishing=False, include_prob_term =True, manager_pickle_file = "man_02_n06.p",
                            prob_feat_select= 0.2, preference_distribution_string="power_law")

