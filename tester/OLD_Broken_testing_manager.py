import numpy as np
import pickle
import math
from manager.MOD_main_manager import Manager
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

print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(" !!!!!!!!!!!!!!!!!!!!!!WE ARE USING MODIFIED MAIN MANAGER see imports!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

def test_full_cycle_and_accuracy(test_size, num_rounds , num_plans_per_round , exploration_rounds , num_dbs_samples,
                                 include_gain = False, random_seed = 40,manager_pickle_file = None,input_rating_noise =0.3):

    learn_LSfit = True
    random_sampling_enabled = True
    print("RANOM SAMPLING ENABLED = ",random_sampling_enabled)
    in_region_error = []
    out_region_error = []
    bayes_error_list = []
    MLE_error_list = []
    try:
        with open(manager_pickle_file,"rb") as src:
            manager = pickle.load(src)
            pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
            pref_list = sorted(pref_list, key=lambda x: x[0])
            print("-=-=-=-= USING PICKLED FILE-=-=-=-=-=-")
    except :
        RATING_NOISE = input_rating_noise
        if exploration_rounds == 0:
            exploration_rounds += 1
        manager = Manager(FEATURE_FREQ_CUTOFF= 0.05, test_set_size=test_size ,prob_per_level=(0.3,0.2),random_seed=random_seed)
        pref_list = [x for x in manager.sim_human.feature_preferences_dict.items()]
        pref_list = sorted(pref_list, key=lambda x: x[0])
        print("TRUE FEATURE PREFERENCE DICT", pref_list)
        print("num features =", len(pref_list))
        print("Include gain is =", include_gain)
        with open(manager_pickle_file, "wb")as dest:
            pickle.dump(manager,dest)

    #end except for when the pickle file is not found

    print("finished initializing manager, starting testing")



    manager.sim_human.change_rating_noise(input_rating_noise)  # SET NOISE IN RATING
    #now sample
    random.seed(random_seed)
    for round_num in range(num_rounds):
        print("============ ROUND ",round_num,"====================")
        if round_num < exploration_rounds:
            if random_sampling_enabled:
                sampled_plans = manager.sample_randomly_wDiversity(num_plans_per_round)
                # sampled_plans = manager.sample_randomly(num_plans_per_round)
            else:
                sampled_plans = manager.sample_by_DBS(num_plans_per_round)


        else:
            #----exploratory sampling
            if random_sampling_enabled:
                sampled_plans = manager.sample_randomly_wDiversity(num_dbs_samples)
                # sampled_plans = manager.sample_randomly(num_dbs_samples)
            else:
                sampled_plans = manager.sample_by_DBS( num_dbs_samples)

            #---also sample by rbus to exploit the features discovered
            if random_sampling_enabled:
                sampled_plans += manager.sample_randomly_wDiversity(num_samples=num_plans_per_round - num_dbs_samples)
                # sampled_plans += manager.sample_for_SAME_features(num_samples=num_plans_per_round - num_dbs_samples)
                # sampled_plans += manager.sample_by_DBS(num_samples=num_plans_per_round - num_dbs_samples)
                # sampled_plans += manager.sample_randomly(num_samples=num_plans_per_round - num_dbs_samples)
            else:
                sampled_plans += manager.sample_by_RBUS(num_samples=num_plans_per_round - num_dbs_samples,include_gain = include_gain)



        annotated_plans = manager.get_feedback(sampled_plans)
        if round_num == num_rounds-1:
            learn_LSfit = True
        #end if
        manager.update_indices(annotated_plans)
        manager.relearn_model(learn_LSfit, num_chains=2) # here is where we first train the model

        test_plans = manager.get_balanced_test_set()  # this is ONLY for every round evaluation. The last round should al
        manager.sim_human.change_rating_noise(0.0)  # todo NOTE the test dataset has no noise.
        annotated_test_plans = manager.get_feedback(test_plans)
        manager.test_set = annotated_test_plans
        scores = []
        for plan in annotated_test_plans:
            scores.append(plan[-1])
        print("TEST set scores stats")
        print(summ_stats_fnc(scores))
        manager.sim_human.change_rating_noise(input_rating_noise)#IMPORTANT TO RETURN THE RATING TO NOISY

        bayes_error, MLE_error = manager.evaluate(annotated_test_plans)
        bayes_error_list.append(bayes_error)
        MLE_error_list.append(MLE_error)
        in_region_error.append(manager.region_based_evaluation(annotated_test_plans, [(0.0, 0.2), (0.8, 1.0)],
                                                               ONLY_inside_region=True)) # the second parameter is percentile regions to evaluate in
        out_region_error.append(manager.region_based_evaluation(annotated_test_plans, [(0.0, 0.2), (0.8, 1.0)],
                                                                ONLY_inside_region=False)) # the second parameter is percentile regions to evaluate in
        #TODO VERY IMPORTANT TO UPDATE INDICES AFTER THE EVALUATION, else all the weights are not correctly learned
        # todo analyse the learnt model to see if all the MCMC chains are agreeing with each other
    #end for loop through
    #---now measure the accuracy


    # currently we are testing with gain  = 1, and only variance.
    # ALSO increase noise and compare performance. The difference in region and not region is expected to be more.
    # todo analyse the learnt model to see if all the MCMC chains are agreeing with each other

    # manager.learning_model.plot_learnt_parameters()
    pref_dict = dict(pref_list)
    print("Include gain is =", include_gain)
    print("TRUE FEATURE PREFERENCE", pref_list)
    print("num features =", len(pref_list), "gain included is ",include_gain)
    print("FEATURES DISCOVERED ", manager.relevant_features_dimension , " ", [(x,pref_dict[x]) for x in manager.relevant_features])
    print("IN REGION ERROR =" , in_region_error)
    print("OUT OF REGION ERROR =" ,out_region_error)
    print("============================================================")
    print("BAYES ERROR LIST ", bayes_error_list)
    print("MLE ERROR LIST ", MLE_error_list)
    print("MIN,MAX",manager.min_rating,manager.max_rating)

    #todo remove plotting code for speed
    #manager.learning_model_bayes.plot_learnt_parameters()
    print("RANOM SAMPLING ENABLED = ",random_sampling_enabled)

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

if __name__ == "__main__":

    # manager_pickle_file = "UNIFORM_01_01_v1_8_noise00.p"
    # with open(manager_pickle_file, "rb") as src:
    #     manager = pickle.load(src)
    #     annotated_test_plans = pickle.load(src)
    #
    #
    # #annotate all the data points
    # all_data_annotated = manager.get_feedback(manager.formatted_plan_dataset)
    # all_data_annotated = all_data_annotated[:]
    # annotated_test_plans = all_data_annotated
    # num_points = len(annotated_test_plans)
    # manager.RBUS_indexing = sorted(list(manager.relevant_features))
    # manager.RBUS_prior_weights = [0.0] * manager.relevant_features_dimension
    # coefficients = np.array([manager.sim_human.feature_preferences_dict[x][-1] for x in manager.RBUS_indexing])
    # #BAD reduce the number of coefficients
    # # coefficients = coefficients*np.array([1.0]*int(manager.relevant_features_dimension/2) + [0.0]*int(manager.relevant_features_dimension - manager.relevant_features_dimension/2))
    # print(coefficients)
    # all_encoded_plans = []
    # for single_annot_plan_struct in annotated_test_plans:
    #     current_plan_features = single_annot_plan_struct[2] + single_annot_plan_struct[3]
    #     encoded_plan = np.zeros(manager.relevant_features_dimension)
    #     for single_feature in current_plan_features:
    #         if single_feature in manager.relevant_features:
    #             encoded_plan[manager.RBUS_indexing.index(single_feature)] = 1
    #     #end inner for
    #     all_encoded_plans.append(encoded_plan)
    # # end outer for
    # all_encoded_plans = np.array(all_encoded_plans)
    # column_sums = np.sum(all_encoded_plans,axis=0)
    # print(list(zip(list(column_sums),list(range(manager.relevant_features_dimension)))))
    #
    # # num_points = 1000
    # # num_dimens = 5
    # # coeff_lowerbound = -3
    # # coeff_upperbound =4
    # # toy_data_input = np.random.randint(low=0,high=2,size=(num_points,num_dimens))#high of 2, means max is high-1 = 1.
    # # # toy_data_linear_coeff = np.array([2,3,-1,0,0]) #later we could do random for this too
    # # toy_data_linear_coeff = np.random.randint(low = coeff_lowerbound,high=coeff_upperbound,size=num_dimens)
    # # print(toy_data_linear_coeff)
    # # toy_data_output = np.zeros(num_points) #then compute the output
    #
    # comparison_data = np.sum(np.tile(coefficients,(num_points,1)) * all_encoded_plans, axis=1)
    # toy_data_output = comparison_data
    # # toy_data_output = [x[-1] for x in annotated_test_plans]
    # # print(np.mean(toy_data_output-comparison_data),np.var(toy_data_output-comparison_data))
    # learned_coeff,learned_offset,manager.learning_model_bayes.linear_params_values = test_bayesian_MV_linModel(all_encoded_plans,toy_data_output)
    # print("Offset ", learned_offset)
    # error = np.array(coefficients) - np.array(learned_coeff)
    # print(error)
    # print(np.mean(error))
    # print(np.var(error))
    #
    # #----actual evaluation on the learned params
    # bayes_total_squared_error = 0.0
    # bayes_error_list = []
    # count_samples = 0
    # for encoded_plan,rating in zip(all_encoded_plans,toy_data_output):
    #     if rating == 0.0:
    #         continue
    #     count_samples += 1
    #     predictions, kernel = manager.learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan,
    #                                                                                           num_samples=500)
    #     preference_possible_values = np.linspace(-1.0,1.0, num=100)
    #     preference_prob_density = kernel(preference_possible_values)
    #     index_mode = np.where(preference_prob_density == np.max(preference_prob_density))[0][0]
    #     mode_prediction = preference_possible_values[index_mode]
    #     current_squared_error = math.pow(rating - mode_prediction, 2)
    #     print(rating,mode_prediction)
    #     bayes_total_squared_error += current_squared_error
    #     bayes_error_list.append(math.sqrt(current_squared_error))
    #
    # bayes_final_error = math.sqrt(bayes_total_squared_error / count_samples)
    # print("BAYES MODEL The average error in ALL regions= ", bayes_final_error)
    # print("BAYES MODEL Error Statistics of ALL regions , ", summ_stats_fnc(bayes_error_list))

#=================================END TOY DATA TESTING

    #todo
    # do not look at the mean, look at the mean and variance for each parameter !! mean alone is misleading and maybe spread over
    #     a large range !! try different sigma(sd) and see how it changes


    total_num_plans = 120
    plans_per_round = 30
    random_seed = 150
    noise_value = 1.0
    #todo remove all features known in "manager"
    manager_pickle_file = "MAN_noise10_set4.p"
    print(manager_pickle_file)
    # manager_pickle_file = "Delete1.p"
    for i in range(1):
        with CodeTimer():
            NOgain_in_region_error,NOgain_out_region_error, NOgain_bayes_error_list,NOgain_MLE_error_list = \
                test_full_cycle_and_accuracy(test_size=1000, num_rounds = int(total_num_plans/plans_per_round),
                    num_plans_per_round = plans_per_round,exploration_rounds= int(0.33 * total_num_plans/plans_per_round),num_dbs_samples=int(plans_per_round/5)
                                         , include_gain=False, random_seed= random_seed, manager_pickle_file=manager_pickle_file,input_rating_noise=noise_value)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for i in range(1):
        with CodeTimer():
            in_region_error, out_region_error, bayes_error_list, MLE_error_list = \
                test_full_cycle_and_accuracy(test_size=1000,num_rounds = int(total_num_plans/plans_per_round),
                    num_plans_per_round = plans_per_round,exploration_rounds= int(0.33 * total_num_plans/plans_per_round),num_dbs_samples=int(plans_per_round/5)
                                         , include_gain=True, random_seed= random_seed,manager_pickle_file=manager_pickle_file,input_rating_noise= noise_value)


    x_axis = [plans_per_round*x for x in range(1,int(total_num_plans/plans_per_round)+1)]
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    y_axis = NOgain_MLE_error_list
    plt.plot(x_axis,y_axis, color = 'r')
    y_axis = NOgain_bayes_error_list
    plt.plot(x_axis, y_axis, color='b')
    #----
    ax2 = fig.add_subplot(2,1,2)
    y_axis = MLE_error_list
    plt.plot(x_axis, y_axis, color='r')
    y_axis = bayes_error_list
    plt.plot(x_axis,y_axis, color = 'b')
    plt.show()