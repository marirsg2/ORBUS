"""

Small batch size
?? weights larger than noise. 0.5 >0.2
gain calculation and sampling is bad.

TODO NOTES:
    Do not test on those points for which you have high variance !!! means you are not confident, so do not predict
    or you can BET on points with your variance. Scored by how many you get right, and the error on them. Count matters.
    Betting score.

    After your variance update was fixed, i think FEATURE DISTINGUISHING. div/F is good !
    the gain should become gain*prob/F or just gain*prob AND THEN NORMALIZE, before multiplying with variance. !!
    The gain is the expected gain or benefit, which is determined by how often it occurs, and /F because of how much
    information you can glean from it. Fewer features is better.

TODO NOTE:
    If there are no clean subsets of equations, then uncertainty sampling will win sometimes. Since once the system is constrained
    it will drop. PUT A LOT OF NOISE, AND THEN SEE HOW IT PERFORMS

todo note:
    have a validation set that tells you when to stop, sometimes, getting MORE data can increase error, since noise from other ratings
    makes it worse ? NONSENSE !! YOUR VALIDATION DATA MAY HAVE NOISE TOO AND THIS IS ACTIVE LEARNING, NO GOLDEN SET OF DATA.



"""

import pickle
import datetime
import numpy as np
import sys
import os
import itertools
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
                                 include_gain = False, include_discovery_term=True, include_feature_distinguishing=True,
                                 include_prob_term = True,include_feature_feedback =True,
                                 random_seed = 40,
                                 manager_pickle_file = None, input_rating_noise =0.2,
                                 prob_feat_select= 0.2, preference_distribution_string="power_law"):

    learn_LSfit = True
    RATIO_TEST_SET = 0.5
    print("doing probability per level =", prob_feat_select)
    print("include_gain = ",include_gain)
    print("include_discovery_term = ",include_discovery_term)
    print("include_feature_distinguishing = ",include_feature_distinguishing)
    print("include_prob_term = ",include_prob_term)
    print("include_feature_feedback = ",include_feature_feedback)
    print("RANOM SAMPLING ENABLED = ", random_sampling_enabled)
    print("Rating Noise =", input_rating_noise)
    print("RATIO TEST SET =", RATIO_TEST_SET)
    in_region_error = []
    bayes_error_list = []
    MLE_error_list = []
    inRegion_bayes_error_list = []
    UNALTERED_inRegion_bayes_error_list = []
    inRegion_MLE_error_list = []
    annotated_test_plans = []

    #todo NOTE this.
    if random_sampling_enabled == True:
        random_seed = random.randint(1,100)

    try:
        with open(manager_pickle_file,"rb") as src:
            manager = pickle.load(src)
            annotated_test_plans = pickle.load(src)
            pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
            pref_list = sorted(pref_list, key=lambda x: x[0])
            manager.set_AL_params(use_feature_feedback = include_feature_feedback,random_seed=random_seed)
            print("TRUE FEATURE PREFERENCE", pref_list)
            freq_info = sorted(manager.freq_dict.items(),key=lambda x:x[1])
            print("FREQ INFO", freq_info)
            sorted_freq_pref_list = [(x[0], manager.freq_dict[x[0]], x[1]) for x in pref_list]
            sorted_freq_pref_list = sorted(sorted_freq_pref_list,key= lambda x:x[-1][-1])
            print("PREF & FREQ = ",sorted_freq_pref_list)
            print("num features =", len(pref_list))
            print("-=-=-=-= USING PICKLED FILE-=-=-=-=-=-")
    except :
        print("RECREATE manager")
        manager = Manager(prob_feature_selection=prob_feat_select,use_feature_feedback = include_feature_feedback,
                          random_seed=random_seed, preference_distribution_string=preference_distribution_string,
                          preference_gaussian_noise_sd=input_rating_noise)
        pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
        pref_list = sorted(pref_list, key=lambda x: x[0])
        print("TRUE FEATURE PREFERENCE", pref_list)
        freq_info = sorted(manager.freq_dict.items(), key=lambda x: x[1])
        print("FREQ INFO", freq_info)
        sorted_freq_pref_list = [(x[0], manager.freq_dict[x[0]], x[1]) for x in pref_list]
        sorted_freq_pref_list = sorted(sorted_freq_pref_list, key=lambda x: x[-1][-1])
        print("PREF & FREQ by pref = ", sorted_freq_pref_list)
        sorted_freq_pref_list = [(x[0], manager.freq_dict[x[0]], x[1]) for x in pref_list]
        sorted_freq_pref_list = sorted(sorted_freq_pref_list, key=lambda x: x[1])
        print("PREF & FREQ by freq = ", sorted_freq_pref_list)
        print("num features =", len(pref_list))
        print("Include gain is =", include_gain)
        manager.sim_human.change_rating_noise(0.0)# todo NOTE the test dataset has no noise.
        test_plans = manager.extract_test_set(test_set_size= int(RATIO_TEST_SET*len(manager.plan_dataset)))
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
    print("NUM PLANS =", len(manager.plan_dataset))
    # print(sorted(scores))

    manager.sim_human.change_rating_noise(input_rating_noise)  # SET NOISE IN RATING

    random.seed(random_seed)

    for round_num in range(1,num_rounds+1):
        print("============ ROUND ",round_num,"====================")
        # if round_num == 1:
        #     sampled_plans = manager.sample_randomly(num_plans_per_round)
        # else:
        if random_sampling_enabled:
            sampled_plans = manager.sample_randomly(num_plans_per_round)
        else:
            sampled_plans = manager.IMPORTANT_get_plans_for_round(num_plans_per_round, use_gain_function=include_gain, \
                                                                  include_feature_distinguishing= include_feature_distinguishing, \
                                                                  include_discovery_term_product=include_discovery_term,
                                                                  include_probability_term = include_prob_term)

        annotated_plans = manager.get_feedback(sampled_plans)
        print("FEEDBACK AND QUERIES FOR ROUND ", round_num, " are \n", annotated_plans)
        print("SUM OF PROB OF SEEN FEATURES =", sum([manager.freq_dict[x] for x in manager.seen_features]) ," ; SEEN FEATURES = ", manager.seen_features)
        print("LIKED FEATURES ", manager.liked_features)
        print("DISLIKED FEATURES ", manager.disliked_features)
        print("RELEVANT FEATURES PROB MASS = ",sum([manager.freq_dict[x] for x in manager.liked_features.union(manager.disliked_features)]) )
        print("RELEVANT FEATURES PREF MASS = ",sum([abs(manager.sim_human.feature_preferences_dict[x][-1]) for x in manager.liked_features.union(manager.disliked_features)]) )
        print("UNKNOWN FEATURES ", manager.all_s1_features.difference(manager.seen_features))
        print("MIN,MAX", manager.min_rating, manager.max_rating)
        print("ANNOTATED MIN,MAX", manager.annotated_min, manager.annotated_max)
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
        manager.relearn_model(learn_LSfit, num_chains=1) # here is where we first train the model
        #todo REMOVE THIS and maybe move it to replace the test set
        # manager.select_best_and_worst(30)
        bayes_error, MLE_error = manager.evaluate(annotated_test_plans)
        bayes_error_list.append(bayes_error)
        MLE_error_list.append(MLE_error)
        inRegion_bayes_error,UNALTERED_inRegion_bayes_error, inRegion_MLE_error,inRegion_true_values_and_diff = manager.region_based_evaluation(annotated_test_plans, [0.2,0.8],
                                        inside_region=True)  # the second parameter is percentile regions to evaluate in
        inRegion_bayes_error_list.append(inRegion_bayes_error)
        UNALTERED_inRegion_bayes_error_list.append(UNALTERED_inRegion_bayes_error)
        inRegion_MLE_error_list.append(inRegion_MLE_error)
        inRegion_true_values_and_diff = sorted(inRegion_true_values_and_diff,key=lambda x:x[0])
        print("INTERESTING REGION TRUE VALUES ", inRegion_true_values_and_diff)
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
    print("FREQ DICT", sorted(manager.freq_dict.items(),key = lambda x:x[1]))
    print("PREF & FREQ = ", [(x[0],manager.freq_dict[x[0]],x[1]) for x in pref_list])
    print("num features =", len(pref_list), "gain included is ",include_gain)
    try:
        print("FEATURES DISCOVERED ", manager.POSSIBLE_features_dimension, " ", [(x, pref_dict[x]) for x in manager.POSSIBLE_features])
    except: #when we do not use features, it will throw an error
        pass
    print("ANNOTATED PLANS BY ROUND =", manager.annotated_plans_by_round)
    print("IN REGION ERROR =" , in_region_error)
    print("============================================================")
    print("BAYES ERROR LIST ", bayes_error_list)
    print("MLE ERROR LIST ", MLE_error_list)
    print("INTERESTING REGION TRUE VALUES ", inRegion_true_values_and_diff)
    print("INTERESTING REGION BAYES ERROR LIST ", inRegion_bayes_error_list)
    print("---UNALTERED  INTERESTING REGION BAYES ERROR LIST ", UNALTERED_inRegion_bayes_error_list)
    # print("INTERESTING REGION  MLE ERROR LIST ", inRegion_MLE_error_list)
    print("MIN,MAX",manager.min_rating,manager.max_rating)
    print("ANNOTATED MIN,MAX",manager.annotated_min,manager.annotated_max)

    #todo remove plotting code for speed
    #manager.learning_model_bayes.plot_learnt_parameters()
    print("manager file =", manager_pickle_file)
    print("include_gain = ",include_gain)
    print("include_discovery_term = ",include_discovery_term)
    print("include_feature_distinguishing = ",include_feature_distinguishing)
    print("include_prob_term = ",include_prob_term)
    print("include_feature_feedback = ",include_feature_feedback)
    print("RANOM SAMPLING ENABLED = ", random_sampling_enabled)
    print("Rating Noise =", input_rating_noise)
    print("used Indices (testplans+annot) =", manager.indices_used)
    print("--------END OF TESTING-------")

    return (bayes_error_list,MLE_error_list,inRegion_bayes_error_list, inRegion_MLE_error_list,[manager.min_rating,manager.max_rating])
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


def Active_Learning_Testing(total_num_plans = 240, plans_per_round = 30, random_seed = 150, noise_value = 0.2, random_sampling_enabled = False,
                            include_gain=True,include_discovery_term=True, include_feature_distinguishing=True, include_prob_term=True,  include_feature_feedback=True,
                            manager_pickle_file = "default_man.p",
                            prob_feat_select= 0.2, preference_distribution_string="power_law",repetitions = 1):


    print ("doing probability per level =", prob_feat_select)
    print(manager_pickle_file)
    ret_struct = []
    for i in range(repetitions):
        with CodeTimer():
             ret_struct.append(test_full_cycle_and_accuracy(test_size=1000, num_rounds = int(total_num_plans/plans_per_round),
                                             num_plans_per_round = plans_per_round,
                                             random_sampling_enabled = random_sampling_enabled, include_gain=include_gain,
                                                            include_discovery_term=include_discovery_term,include_feature_distinguishing=include_feature_distinguishing,
                                             include_prob_term = include_prob_term, include_feature_feedback = include_feature_feedback,
                                             random_seed= random_seed, manager_pickle_file=manager_pickle_file, input_rating_noise=noise_value,
                                             prob_feat_select=prob_feat_select, preference_distribution_string=preference_distribution_string))
    #end for loop
    return ret_struct



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



    all_data = []
    num_repetitions = 1
    NUM_RANDOM_SAMPLES = 5
    num_parameters = 4
    parameter_values = [True, False]
    parameter_indexed_values = [parameter_values] * num_parameters
    cases = itertools.product(*parameter_indexed_values)

    # preference_distribution_string = "uniform"
    # preference_distribution_string = "power_law"
    # preference_distribution_string = "gaussian"
    preference_distribution_string = "gumbel"
    total_num_plans = 30
    #TODO FEWER PLANS PER ROUND IS BETTER
    plans_per_round = 2
    noise_value = 0.2 #the range of actual preference values is based on the noise as well
    prob_feat_select = 0.4

    # date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    # date_time_str = date_time_str.replace(" ", "_")
    # date_time_str = date_time_str.replace("/", "_")
    # date_time_str = date_time_str.replace(",", "_")
    # date_time_str = date_time_str.replace(":", "_")
    # print("date and time:", date_time_str)
    # output_file_name = 'RBUS_output_results' + "_" + date_time_str
    # sys.stdout = open(output_file_name + '.txt', 'w')
    # print('test')

    manager_pickle_file = "man_02_n06.p"
    # random_seed = 666 #-1 means do not fix randomness. handled in code later


    # CHECK THAT THE MAX VARIANCE IS IN QUERIES WHERE THE SUM OF INDIVIDUAL VARIANCE IS MAX.


    # cases = [list(x) for x in cases]
    # print("The parameter cases are ",cases)


    for i in range(4):
        all_data = []
        try:
            os.remove(manager_pickle_file)
            print("Manager File Removed at start , to recreate manager!")
        except FileNotFoundError:
            pass  # file was already deleted

        random_seed = int(random.randint(1,1000))
        date_time_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        date_time_str = date_time_str.replace(" ", "_")
        date_time_str = date_time_str.replace("/", "_")
        date_time_str = date_time_str.replace(",", "_")
        date_time_str = date_time_str.replace(":", "_")
        print("date and time:", date_time_str)
        output_file_name = 'RBUS_output_results' + "_" + date_time_str + "_CASE_"+str(i)
        sys.stdout = open(output_file_name + '.txt', 'w')
        print('test')
        # random.shuffle(cases)
        # include_discovery_term = case_parameters[0], include_gain = case_parameters[1], include_feature_distinguishing = case_parameters[2],include_prob_term = case_parameters[3],
        # special_order_cases = [[True, True, False, True], [True, False, False, True], [True, True, True, True], [True, False, True, True]]#,[True, True, True, True],[True, False, True, True],[True, True, False, False]]
        # special_order_cases = [[True, True, False, True], [False, True, False, True], [True, True, False, False], [False, True, False, False]]# only the gain term is kept, all else is changed. The feature distinguishing is kept to False
        special_order_cases = [[True, True, False, True], [False, True, False, False]]# only the gain term is kept, all else is changed. The feature distinguishing is kept to False
        # special_order_cases = [[True, True, False, True], [True, False, False, True]]#, [True, True, True, True], [True, False, True, True]]#,[True, True, True, True],[True, False, True, True],[True, True, False, False]]
        cases = special_order_cases

        print(" ALWAYS CHECK THE FOLLOWING PARAMETERS ")
        print(" What technique , priors, distribution, dataset")

        # --------------------------------------------------------------
        # random_sampling_state = True
        # for i in range(1):
        #     all_data.append(([random_sampling_state]+cases[0], Active_Learning_Testing(total_num_plans = total_num_plans, plans_per_round = plans_per_round, random_seed = random_seed, noise_value = noise_value ,
        #                             random_sampling_enabled =  random_sampling_state,
        #                             include_feature_feedback= True,
        #                             include_discovery_term = False,
        #                             include_gain= False,
        #                             include_feature_distinguishing= False,
        #                             include_prob_term = False,
        #                             manager_pickle_file = manager_pickle_file,
        #                             repetitions=num_repetitions,
        #                             prob_feat_select= prob_feat_select, preference_distribution_string=preference_distribution_string)))


        #--------------------------------------------------------------

        random_sampling_state = False
        for case_parameters in cases:
            all_data.append(([random_sampling_state]+case_parameters, Active_Learning_Testing(total_num_plans = total_num_plans, plans_per_round = plans_per_round, random_seed = random_seed, noise_value = noise_value ,
                                    random_sampling_enabled =  random_sampling_state,
                                    include_feature_feedback= True,
                                    include_discovery_term = case_parameters[0],
                                    include_gain= case_parameters[1],
                                    include_feature_distinguishing= case_parameters[2],
                                    include_prob_term = case_parameters[3],
                                    manager_pickle_file = manager_pickle_file,
                                    repetitions=num_repetitions,
                                    prob_feat_select= prob_feat_select,
                                    preference_distribution_string= preference_distribution_string)))
            single_data_set = all_data[-1]
            case_parameters = single_data_set[0]
            print("============================================================")
            print("CASE DESCRIPTIONS")
            print("|| feature feedback = True \n || random_sampling_enabled =", case_parameters[0],
                  " || include_discovery =", case_parameters[1],
                  " || include_gain =", case_parameters[2],
                  " || include_feature_distinguishing =", case_parameters[3],
                  " || include_prob_term =", case_parameters[4])
            for i in range(num_repetitions):
                print("BAYES ERROR LIST ", single_data_set[1][i][0])
                print("MLE ERROR LIST ", single_data_set[1][i][1])
                print("INTERESTING REGION BAYES ERROR LIST ", single_data_set[1][i][2])
                print("INTERESTING REGION  MLE ERROR LIST ", single_data_set[1][i][3])
                print("MIN,MAX", single_data_set[1][i][4])
                print("============================================================")

        # random_sampling_state = True
        # for i in range(NUM_RANDOM_SAMPLES-1):
        #     all_data.append(([random_sampling_state]+case_parameters, Active_Learning_Testing(total_num_plans = total_num_plans, plans_per_round = plans_per_round, random_seed = random_seed, noise_value = noise_value ,
        #                             random_sampling_enabled =  random_sampling_state,
        #                             include_feature_feedback= True,
        #                             include_discovery_term = False,
        #                             include_gain= False,
        #                             include_feature_distinguishing= False,
        #                             include_prob_term = False,
        #                             manager_pickle_file = manager_pickle_file,
        #                             repetitions=num_repetitions,
        #                             prob_feat_select= prob_feat_select, preference_distribution_string=preference_distribution_string)))
        #end for loop through the cases and collecting data
        print("============================================================")
        print(all_data)
        print("============================================================")

        for single_data_set in all_data:
            case_parameters = single_data_set[0]
            print("============================================================")
            print("CASE DESCRIPTIONS")
            print( "|| feature feedback = True \n || random_sampling_enabled =", case_parameters[0],
                    " || include_discovery =", case_parameters[1],
                    " || include_gain =", case_parameters[2],
                    " || include_feature_distinguishing =", case_parameters[3],
                    " || include_prob_term =", case_parameters[4])
            for i in range(num_repetitions):
                print("BAYES ERROR LIST ", single_data_set[1][i][0])
                print("MLE ERROR LIST ", single_data_set[1][i][1])
                print("INTERESTING REGION BAYES ERROR LIST ", single_data_set[1][i][2])
                print("INTERESTING REGION  MLE ERROR LIST ", single_data_set[1][i][3])
                print("MIN,MAX", single_data_set[1][i][4])
                print("============================================================")

        #END FOR LOOP through printing the data
        sys.stdout.flush()
    #END for loop through repeating
    sys.stdout.flush()