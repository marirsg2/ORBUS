
"""
search for "todo index"
that is where code changes are noted

#todo compare technique 2 and 2_v2. Search for keyword "technique"

#todo note: this maybe a shit bad idea, and too complex to do elegantly. very specific and hackish.
# FIND SUBSETS of features to complete. does div /|F| implicitly do that !!! ??
#there may NOT exist a subset of plans that neatly wraps up the parameters seen



TODO CURRENT SETTING:, MOVE THESE PARAMETERS TO THE TESTER.
    SEE TESTING MANAGER PARAMS
    mean 10, var 5
    g^*var^. norm by div by sigma !! NOT BY MAX


ALWAYS START WITH SIMPLE CASE OF ALL RELEVANT AND KNOWN FEATURES.
HARD CODE ALL S1 FEATUERS AS CONFIRMED, AND SET PROB OF RELEVANCE AS 1.0

"""
from itertools import combinations
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import timedcall, metric
from scipy import integrate
from scipy.stats import describe as summ_stats_fnc
#---------------------------------------------------------
from plan_generator.journey_world import Journey_World
from plan_generator.journey_world_planner import Journey_World_Planner
from feedback.oracle import oracle
from multiprocessing import Pool
import numpy as np
import pickle
import random
import copy
import math
from scipy.stats import norm
from linetimer import CodeTimer
# from common.system_logger.general_logger import logger
from learning_engine.bayesian_linear_model import bayesian_linear_model
from selection_engine import RBUS_selection
import re

"""
IMPORTANT NOTES:

SEE ALL TODO NOTES

0) ! I dont think we need to use the variance AND the gain in RBUS sampling. The gain score alone HAS the variance implicitly 
considered. When we multiplied the gain function sample points with the prob density function of the predicted output, we factor
in the variance

1) We parallelized the RBUS integration step
2) WE hard drop those plans that have the same encoding as any of the previous plans. THIS MAKES SENSE.
if they were high freq features that were different, they would not have the same encoding. ALSO if new features come to light
then in a future round they will not have the same encoding
3) SEE the bayes linear model file. for learnings in that area 


"""
#===============================================================

EXPECTED_NOISE_STD_DEV = 0.2  #the mean is 0
NUM_SAMPLES_KDE = 500
NUM_SAMPLES_XAXIS_SAMPLES = 100

INFINITESIMAL_VALUE = 1e-15

#===============================================================

# --------------
def get_biModal_gaussian_gain_function(a=0.2, b=0.8, sd=0.1):
    """
    :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
    There is no special property with probabilities and summing to 1 in this application. Only relative gain values
    that match what we need.
    :param x_value:
    :param a:
    :param b:
    :return: the function
    """
    gain_a = norm(a, sd)
    gain_b = norm(b, sd)
    return lambda x: (gain_a.pdf(x) + gain_b.pdf(x)) / 2


# ================================================================================================

def get_gain_function(min_value, max_value, lower_percentile=0.1, upper_percentile=0.9):
    """
    :summary: abstract gain function
    :param x_value:
    :return:
    :return:
    """
    range = max_value - min_value
    lower_fifth = min_value + range * lower_percentile
    upper_fifth = min_value + range * upper_percentile
    scaled_std_dev = range * 0.1
    return get_biModal_gaussian_gain_function(lower_fifth, upper_fifth, scaled_std_dev)


# ================================================================================================

def string_char_split(input_string):
    """

    :param input_string:
    :return:
    """
    ret_obj = []
    ret_obj.extend(input_string)
    return ret_obj

#===============================================================

def look_for_ordered_feature(feature_char_list, plan_char_seq):
    """

    :param feature_char_list:
    :param plan_char_seq:
    :return:
    """
    found_state = True
    for single_char in feature_char_list:
        if single_char not in plan_char_seq:
            found_state = False
            break
        #end if
        plan_char_seq = plan_char_seq[plan_char_seq.index(single_char)+1:]
    #end for
    return found_state


#===============================================================
def insert_liked_disliked_features(all_plans,liked_features,disliked_features):
    """

    :param all_plans:
    :param liked_features:
    :param disliked_features:
    :return:
    """
    #todo NOTE cannot just check the features in the formatted dict because rare features below cutoff will not be there
    formatted_plans_struct = []
    for single_plan in all_plans:
        found_like_features = []
        for single_liked in liked_features:
            if look_for_ordered_feature(single_liked,single_plan):
                found_like_features.append(string_char_split(single_liked))
        found_dislike_features = []
        for single_disliked in disliked_features:
            if look_for_ordered_feature(single_disliked,single_plan):
                found_dislike_features.append(string_char_split(single_disliked))
        formatted_plans_struct.append( [single_plan,found_like_features,found_dislike_features] )
    #end outermost for
    return formatted_plans_struct


#===============================================================

def planTrace_clustering(start_medoids, data_points, tolerance=0.25, show=False):
    """
    :param start_medoids: is a list of indices, which correspond to the medoids
    :param data_points:
    :param tolerance:
    :param show:
    :return:
    """

    # todo note start mediods is represented by the data point index
    chosen_metric = metric.distance_metric(metric.type_metric.MANHATTAN)
    kmedoids_instance = kmedoids(data_points, start_medoids, tolerance, metric=chosen_metric)
    (ticks, result) = timedcall(kmedoids_instance.process)
    print("execution time in ticks = ",ticks)

    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()

    print("----finished clustering")

    # if show is True:
    #     visualizer = cluster_visualizer_multidim()
    #     visualizer.append_clusters(clusters, data_points, 0)
    #     visualizer.append_cluster([data_points[index] for index in start_medoids], marker='*', markersize=15)
    #     visualizer.append_cluster(medoids, data=data_points, marker='*', markersize=15)
    #     visualizer.show()

    return clusters,medoids

#===============================================================

class Manager:
    """
    The Manager is tasked with handling the whole pipeline. It starts with creating the domain, generating the set of plans,
    applying DBS, then taking feedback from oracle, then applying R-BUS and DBS. Then conducting further rounds. It will
    take care of creating the appropriate data structures for each stage. If we want to recreate a run, then we can the same
    seed.
    """
    RATIO_PLANS_EXPLORATION = 0.2
    def __init__(self,
                 num_rounds = 2,
                 num_dataset_plans = 10000,
                 with_simulated_human = True,
                 max_feature_size = 5,
                 feature_freq_dict = None,  #use this later when there are more involved PGM to compute prob of plan
                 test_set_size=1000,
                 plans_per_round = 30,
                 include_feature_distinguishing=True,
                 use_features_seen_in_plan_dataset = True,
                 prob_feature_selection = 0.25,  #there is ONLY ONE LEVEL, p(like/dislike)
                 pickle_file_for_plans_pool = "default_plans_pool.p",
                 use_feature_feedback = True,
                 relevant_features_prior_mean = (-4, 4), #TODO USE THE AVERAGE WEIGHT AND VARIANCE FOR THESE PARAMS.
                 relevant_features_prior_var = (3, 3),# NOTE THIS IS VAR, WE NEED 2*STD_DEV TO  still positive TODO NOTE DO NOT PUT PRIOR VARIANCE AS NEGATIVE
                 preference_distribution_string="gaussian",
                 preference_gaussian_noise_sd = 0.2,
                 random_seed = 18):
        """
        creates the domain, plan_generator, and the oracle.
        :param num_rounds: for AL
        :param num_dataset_plans: number of backlog plans
        :param with_simulated_human: Or real human(for later use)
        :param relevant_features_prior_var: prior weights set up for liked and disliked features, first element for the
        liked feature
        """

        print("Experiment running with disitribution, mean and priors as ", preference_distribution_string, relevant_features_prior_mean, relevant_features_prior_var)
        self.completed = False
        self.plans_per_round = plans_per_round #plans_per_roud used for the web interface
        self.num_backlog_plans = num_dataset_plans
        self.curr_round = 0 # -1th round is for the test set , 0th round is exploratory
        self.num_rounds = (num_rounds-1)*2 #every even round is RBUS, 0 is DBS
        self.max_feature_size = max_feature_size
        self.learning_model_bayes = bayesian_linear_model()
        self.model_MLE = None # will be set later
        self.use_feature_feedback = use_feature_feedback
        self.like_dislike_prior_mean = relevant_features_prior_mean
        self.like_dislike_prior_var = relevant_features_prior_var
        # self.min_rating = 1e10 #extreme starting values that will be updated after first round of feedback.
        # self.max_rating = -1e10

        self.min_rating = -1.5 #extreme starting values that will be updated after first round of feedback.
        self.max_rating = 1.5
        self.annotated_max = 0.0#will be updated with actually user feedback only
        self.annotated_min = 0.0

        self.indices_used = set()
        self.random_seed = random_seed
        self.sim_human = None

        with open(pickle_file_for_plans_pool,"rb") as src:
            self.plan_dataset = pickle.load(src)

        self.all_s1_features = set()
        for plan in self.plan_dataset:
            for feature in plan:
                self.all_s1_features.add(feature)

        self.annotated_plans = []
        self.annotated_plans_by_round = []
        self.sorted_plans = []
        #important data structures
        # maps feature to index in the vector to be used for DBS/RBUS purpose. So take a plan, and encode it using this
        # data structure when you want to do DBS/RBUS
        self.RBUS_indexing = []
        self.seen_features = set()
        self.liked_features = set()
        self.disliked_features = set()
        self.CONFIRMED_features = self.liked_features.union(self.disliked_features)
                    # print("---BIG ERROR-- RIGHT NOW ALL THE FEATURES ARE KNOWN AT THE START")
                    # for single_feature in self.sim_human.feature_preferences_dict.keys():
                    #     entry =  self.sim_human.feature_preferences_dict[single_feature]
                    #     if entry[0] == "like":
                    #         self.liked_features.add(single_feature)
                    #     else:
                    #         self.disliked_features.add(single_feature)
        self.IRrelevant_features = set()
        self.POSSIBLE_features = self.all_s1_features.difference(self.IRrelevant_features)
        # self.CONFIRMED_features_dimension = len(self.POSSIBLE_features)
        self.CONFIRMED_features_dimension = len(self.CONFIRMED_features)

        #todo NOTE THE BLM TRAINS ON ONLY THOSE WHICH ARE ANNOTATED, to make it faster
        # self.RBUS_indexing = sorted(list(self.POSSIBLE_features))
        # self.RBUS_prior_weights = [0.0]*self.CONFIRMED_features_dimension

        self.RBUS_indexing = []
        self.RBUS_prior_mean = []
        self.RBUS_prior_var = []
        self.num_cores_RBUS = 5

        self.tried_indices = set()
        self.freq_dict = None
        self.compute_freq_dict(self.plan_dataset,self.all_s1_features)
        #todo INDEX 1: cleaned up till here
        if with_simulated_human:

            #todo change this sim human to only use s1 features, those are all the features.
            print("SIMULATED HUMAN has probabilities", prob_feature_selection, " AND gaussian noise sd = ", preference_gaussian_noise_sd, "AND priors =", relevant_features_prior_mean)
            self.sim_human = oracle(self.all_s1_features, probability_of_feat_selec=prob_feature_selection,
                                    gaussian_noise_sd=preference_gaussian_noise_sd, seed=random_seed, freq_dict=self.freq_dict,
                                    preference_distribution_string=preference_distribution_string)

        self.test_set = None# Do not extract test set here !! wait until you have a trained model and then pull interesting plans to test
        #the test set should be well balanced between points inside the interesting region, and outside of it
        self.plans_for_subseq_rounds = [] # will be initialized by the subprocess function that handles the manager
        self.results_by_round = []
    # ================================================================================================
    def set_AL_params(self, use_feature_feedback=True,random_seed=0 ):
        """

        :param prob_feature_selection:
        :param use_feature_feedback:
        :param random_seed:
        :return:
        """
        self.use_feature_feedback = use_feature_feedback
        self.random_seed = random_seed
    # ================================================================================================

    def compute_freq_dict(self, plan_dataset, features_set):
        """
        :summary:
        :param plan_dataset: at index 1, must have the sequence of features in the plan
        :param features_set:
        :return:
        """
        dataset_size = len(plan_dataset)
        freq_dict = {x: 0 for x in features_set}
        for plan in plan_dataset:
            for feature in features_set:
                if feature in plan:
                    freq_dict[feature] += 1 / dataset_size  # this will sum up to the freq at the end
            # end for loop through features
        # end for loop through plans
        self.freq_dict = freq_dict

    # ================================================================================================
    def compute_prob_set(self, input_set):
        """
        :summary: prob of the elements in the set occurring together. Marginalized over all others.
        :param input_set:
        :return:
        """

        prob_plan =1.0
        for feat in input_set:
            prob_plan *= self.freq_dict[feat]
        return prob_plan
    # ================================================================================================

    def update_expected_min_and_max_values(self):
        """
        :return:
        """
        #evaluate the training set and get the min and max values

        temp_max = self.annotated_max
        temp_min = self.annotated_min
        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        for single_plan_idx in available_indices:
            single_plan = self.plan_dataset[single_plan_idx]
            encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
            for single_feature in single_plan:
                if single_feature in self.CONFIRMED_features:
                    encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
            #end for loop
            try:
                predictions = self.learning_model_bayes.get_outputs_from_distribution(encoded_plan,
                                                                                      num_samples=NUM_SAMPLES_KDE)
                mean_prediction = np.mean(predictions)
                prediction_variance = np.var(predictions)
                # if mean_prediction + prediction_variance > temp_max:
                #     temp_max = mean_prediction + prediction_variance
                # if mean_prediction - prediction_variance < temp_min:
                #     temp_min = mean_prediction - prediction_variance
                if mean_prediction  > temp_max:
                    temp_max = mean_prediction
                if mean_prediction  < temp_min:
                    temp_min = mean_prediction
            except:
                return  #means we have no features yet to build a model
        #end outer for loop
        #the annotated max and min hold our existing best estimate for the max and min, then we also consider our predictions
        self.max_rating = temp_min + (temp_max-temp_min)*1.2 #expecting slightly larger preferences.
        self.min_rating = temp_min - (temp_max-temp_min)*0.2

    # ================================================================================================

    def compute_discovery_value(self, input_set):
        """
        :param input_set:
        :return:
        NOTE: we assume equal likelihood of a feature being liked or disliked
        TODO: IF  a likelihood of being relevant is given, then that is the probability that should be used in expected
         discovery calculation. as of now we sum up the probabilities. better than doing 1/n averaging which is VERY conservative
        """
        prob_plan =0.0
        for feat in input_set:
            prob_plan += self.freq_dict[feat]
        return prob_plan

    # ================================================================================================
    def IMPORTANT_get_plans_for_round(self, num_plans=30, use_gain_function=True, include_feature_distinguishing= True, include_probability_term = True,
                                      include_discovery_term_product = False):
        """
        :param self:
        :return:
        :NOTE: the overall INFORMATION FUNCTION is  (RBUS + RBUS/|confirmed features| + RBUS*prob(confirmed features in plan) ) (1 + discovery value of unseen features)

        """
        print("ORBUS RUNNING")
        # if len(self.annotated_plans) == 0:
        #     print("ERROR: no plans were annotated properly, just sending diversity sampling, cannot raise exception, show must go on")
        #     return self.sample_randomly(num_samples) #todo add diverse sampling after fixing it
        # if len(self.annotated_plans) < self.relevant_features_dimension:
        #     #todo CRITICAL CHANGE THIS TO sample to find plans that have the most number of seen features
        #     print("IMPORTANT: CHOOSING TO DO RANDOM SAMPLING over RBUS as the number of annotated plans < number of features")
        #     return self.sample_randomly(num_samples)

        #todo NOTE IMPORTANT that the model must be relearned after the data is input, not here


        num_subProcess_to_use = self.num_cores_RBUS
        #VERY IMPORTANT, AFTER THE MODEL IS LEARNED
        self.update_expected_min_and_max_values()
        # gain_function = RBUS_selection.get_gain_function(min_value=self.min_rating,max_value=self.max_rating)
        # TODO remove all the print statements and integral checks, will speed things up considerably
        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        # print("num available points = ",len(available_indices))
        index_value_list = []  # a list of tuples of type (index,value)
        with CodeTimer():
            p = Pool(num_subProcess_to_use)
            all_parallel_params = []

            # check and divide here if a plan has any features seen, or all unseen.
            # where available indices is used below, replace with blm_indices.
            # do gain *uniform distr for the unseen features and take weighted sum.

            # if self.relevant_features_dimension != 0:
            for single_plan_idx in available_indices:
                current_plan = self.plan_dataset[single_plan_idx]
                encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
                for single_feature in current_plan:
                    if single_feature in self.CONFIRMED_features:
                        encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
                #end for through features
                #special case for first round
                if self.min_rating == 0 and self.max_rating == 0:
                    self.min_rating = -1.0
                    self.max_rating = 1.0
                all_parallel_params.append(
                    [self.learning_model_bayes, encoded_plan, self.min_rating, self.max_rating, use_gain_function])
            # end for loop through the available indices
            # Note the last parameter below is a LIST, each entry is for a new process
            results = p.map(self.parallel_variance_computation,
                            all_parallel_params)  # Note the last parameter is a LIST, each entry is for a new process
            # results = [self.parallel_variance_computation(x) for x in all_parallel_params]
            max_mean_pref = 0
            min_mean_pref = 0
            for single_idx_and_result in zip(available_indices, results):
                single_plan_idx = single_idx_and_result[0]
                single_result = single_idx_and_result[1]
                composite_func_integral, preference_variance,preference_mean = single_result
                if preference_mean > max_mean_pref:
                    max_mean_pref = preference_mean
                if preference_mean < min_mean_pref:
                    min_mean_pref = preference_mean
                index_value_list.append((single_plan_idx, composite_func_integral, preference_variance,preference_mean,self.plan_dataset[single_plan_idx].intersection(self.CONFIRMED_features)))
            #now for those plans that did not have any known relevant features
        # end codetimer profiling section
        # ---NOW we have to select top n plans such that every successive plan selected also considers diversity w.r.t to the previous plans selected
        # the best plan is determined by normalized gain * normalized variance

        #--------TECHNIQUE 1 ---- VAR + gain_norm*VAR
        # print("Using TECHNIQUE 1 ")
        #TERRIBLE AVOID
        # if use_gain_function:
        #     gain_array = np.array([x[1] for x in index_value_list])
        # else:
        #     gain_array = np.array([1.0 for x in index_value_list])
        # # gain_normalizing_denom = np.var(gain_array)
        # gain_normalizing_denom = np.max(gain_array) #so variance is the key factor
        # if gain_normalizing_denom == 0.0:
        #     gain_normalizing_denom = 1.0  # avoids "nan" problem
        # norm_gain_array = gain_array / gain_normalizing_denom  # normalize it
        # variance_array = np.array([x[2] for x in index_value_list])
        # #todo NOTE we multiply by the normalized gain because the defining term is the variance, not the gain.
        # # score = var + var*gain
        # base_score = [variance_array[x] + norm_gain_array[x] * variance_array[x] for x in range(len(variance_array))]
        # unaltered_gain_array = list(copy.deepcopy(gain_array))
        # unaltered_variance_array = list(copy.deepcopy(variance_array))
        # unaltered_basescore_array = list(copy.deepcopy(variance_array))
        # unaltered_meanPref_array = [x[3] for x in index_value_list]


        #TODO try THIS for tech 2. The sqrt is taken so gain is not such a dominating factor. Variance is important, more important. The gain is just to bias it
        # norm_gain_array = np.sqrt(norm_gain_array)
        #---- TECHNIQUE 2---- norm_var * norm_gain, max_norm
        print("Using TECHNIQUE 2 ")
        #todo UNCOMMENT THE PROB AND FEATURE TERMS FURTHER BELOW
        if use_gain_function:
            gain_array = np.array([x[1] for x in index_value_list])
        else:
            gain_array = np.array([1.0 for x in index_value_list])

        # gain_normalizing_denom =  np.max(gain_array)
        gain_normalizing_denom =  np.var(gain_array)

        if gain_normalizing_denom == 0.0:
            gain_normalizing_denom = 1.0  # avoids "nan" problem
        norm_gain_array = gain_array / gain_normalizing_denom  # normalize it

        #TODO I changed this to sqrt to make it about std dev. and set the exponent to 1
        variance_array = np.array([math.sqrt(x[2]) for x in index_value_list])
        # variance_array = np.array([x[2] for x in index_value_list])
        # var_normalizing_denom = np.var(variance_array)
        # var_normalizing_denom = 1.0 #Let variance be the defining factor

        var_normalizing_denom = np.max(variance_array)
        # var_normalizing_denom = np.var(variance_array)

        # OFCOURSE the difference will be small when you have mostly irrelevant features. The number of plans with high weight
        # is smaller, and the overall plan value is smaller in general. So the weighted loss will not be as large as when all
        # the features are relevant
        # gain is var norm, std is max norm
        # currently SMALLER BETA = 1.0 FOR LARGER WEIGHTS AND SEE THE PERFORMANCE, THE NOISE 0.2
        # more FEATURES PER PLAN. the scores should be higher. compare to previous graphs
        #     100 features, ~80 relevant . and 8
        # FEATURES PER PLAN. WHY IS 20 FEATURES NOT WORKING ? GETTING A CONSISTENT ERROR


        #todo need to generate a new DISTRIBUTION Of plans AND SEE if the results persist. Or is this a quaint correlation.
        # NEED TO SAVE THE OLD One, for comparison in case that is true
        # TODO note: SAVE THE STYLE OF NORMALIZATION THAT WORKS BEST !! I think gain/var * std_dev/max seemed to be it. But for simplicity, divide by VAR for both


        exponent = 1
        if var_normalizing_denom == 0.0:
            var_normalizing_denom = 1.0  # avoids "nan" problem
        norm_variance_array = variance_array / var_normalizing_denom  # normalize it
        norm_variance_array = np.power(norm_variance_array, exponent)
        base_score = [norm_gain_array[x]*norm_variance_array[x] for x in range(len(norm_gain_array))]
        unaltered_gain_array = list(copy.deepcopy(gain_array))
        unaltered_variance_array = list(copy.deepcopy(variance_array))
        unaltered_basescore_array = list(copy.deepcopy(variance_array))
        unaltered_meanPref_array = [x[3] for x in index_value_list]

        # ---- TECHNIQUE 2_v2---- norm_var * norm_gain, max_norm
        # print("Using TECHNIQUE 2 v2 ")
        # if use_gain_function:
        #     gain_array = np.array([x[1] for x in index_value_list])
        # else:
        #     gain_array = np.array([1.0 for x in index_value_list])
        # if include_feature_distinguishing:
        #     # += score/|features| # FEATURES THAT ARE CONFIRMED TO BE RELEVANT TO THE USER'S PREFERENCE !!
        #     for x in range(len(index_value_list)):
        #         if len(index_value_list[x][-1]) > 0: #there are features in this plan, some plans have no discovered relevant features. This check avoids div by 0
        #             gain_array[x] = gain_array[x] / (len(index_value_list[x][-1]))
        # if include_probability_term:
        #     # += score* (probabilityOF CONFIRMED FEATURES only)
        #     gain_array = [gain_array[x] *  self.compute_prob_set(index_value_list[x][-1]) for x in range(len(index_value_list))]
        # gain_normalizing_denom = np.max(gain_array)
        # if gain_normalizing_denom == 0.0:
        #     gain_normalizing_denom = 1.0  # avoids "nan" problem
        # norm_gain_array = gain_array / gain_normalizing_denom  # normalize it
        #
        # variance_array = np.array([x[2] for x in index_value_list])
        # var_normalizing_denom = np.max(variance_array)
        # # var_normalizing_denom = 1.0 #Let variance be the defining factor
        # if var_normalizing_denom == 0.0:
        #     var_normalizing_denom = 1.0  # avoids "nan" problem
        # norm_variance_array = variance_array / var_normalizing_denom  # normalize it
        # base_score = [norm_gain_array[x] * norm_variance_array[x] for x in range(len(norm_gain_array))]
        # unaltered_gain_array = list(copy.deepcopy(gain_array))
        # unaltered_variance_array = list(copy.deepcopy(variance_array))
        # unaltered_basescore_array = list(copy.deepcopy(variance_array))
        # unaltered_meanPref_array = [x[3] for x in index_value_list]


        # # ---- TECHNIQUE 3---- CUTOFF in the extreme regions and then use variance
        # variance_array = np.array([x[2] for x in index_value_list])
        # base_score =  variance_array
        #-----------------------------

        # now store (idx,norm_gain*norm_variance)
        addendum = [0.0]*len(index_value_list)
        if include_feature_distinguishing:
            # += score/|features| # FEATURES THAT ARE CONFIRMED TO BE RELEVANT TO THE USER'S PREFERENCE !!
            for x in range(len(index_value_list)):
                if len(index_value_list[x][-1]) > 0: #there are features in this plan, some plans have no discovered relevant features. This check avoids div by 0
                    addendum[x] = addendum[x] + base_score[x] / (len(index_value_list[x][-1]))
        if include_probability_term:
            # += score* (probabilityOF CONFIRMED FEATURES only)
            addendum = [addendum[x] + base_score[x] * self.compute_prob_set(index_value_list[x][-1]) for x in range(len(index_value_list))]
        ## if include_feature_distinguishing:
        ## DO NOT USE THIS METHOD, IT PENALIZES A PLAN WITH ONE FEAUTRE AS 2 FEATURES SINCE YOU +1
        ##     # += score/|features| # FEATURES THAT ARE CONFIRMED TO BE RELEVANT TO THE USER'S PREFERENCE !!
        ##     # todo NOTE this version below does score+score/numFeatures. do NOT need abs because it is always a +ve value. score is an integral for a function to always above zero
        ##     addendum = [addendum[x] + base_score[x]/(1+len(index_value_list[x][-1])) for x in range(len(index_value_list))] # we div by (1+.) to avoid divide by zero error

        if include_discovery_term_product:
            # discovery value is thought of as follows. IF there are two features of marginal probability 0.1, 0.15, then value of discov = 0.25
            # it is the upper bound of coverage of plans in which a feature might appear in. BIAS TO MORE DISCOVERY. could also use expected value, but not done
            discovery_values = [1 + self.compute_discovery_value(self.plan_dataset[index_value_list[x][0]].difference(self.seen_features))
                for x in range(len(index_value_list))]
        index_value_list = [(index_value_list[x][0], base_score[x]+addendum[x]) for x in range(len(index_value_list))]

        UNSCALED_index_value_list = copy.deepcopy(index_value_list)
        # NOTE the order of ENTRIES in index value list will now be fixed
        # see the use of indices list a little further down in code.
        chosen_indices = []
        chosen_scores = []

        # if use_gain_function: #BY TECHNIQUE 3, just sample in this extreme regions only
        #     UNSCALED_index_value_list = []#copy.deepcopy(index_value_list)
        #     temp = []
        #     range_pref_values = max_mean_pref-min_mean_pref
        #     cutoffs = [0.2*range_pref_values+min_mean_pref,0.8*range_pref_values+min_mean_pref]
        #     for i in range(len(index_value_list)):
        #         curr_pref = index_value_list[i][2]
        #         if not (cutoffs[0] < curr_pref and curr_pref < cutoffs[1]):
        #             temp.append(index_value_list[i])
        #             UNSCALED_index_value_list.append(index_value_list[i])
        #     #end for
        #     index_value_list = temp
        # #end if use_gain_function
        chosen_plan_stats = []
        while len(chosen_indices) < num_plans:
            if include_discovery_term_product:
                index_value_list = [
                    (index_value_list[x][0], index_value_list[x][1] * discovery_values[x]) for x in
                    range(len(index_value_list))]
            a = max(index_value_list,key=lambda x:x[1])
            a_idx = index_value_list.index(a)
            chosen_indices.append(a[0])
            chosen_scores.append(a[1])
            feat_value = []
            sum_score = 0.0
            for feat in self.plan_dataset[a[0]]:
                try:
                    feat_value.append((feat,self.sim_human.feature_preferences_dict[feat]))
                    sum_score += self.sim_human.feature_preferences_dict[feat][0]
                except:
                    pass #feature not relevant
            chosen_plan_stats.append((feat_value,sum_score,unaltered_meanPref_array[a_idx], a[1], UNSCALED_index_value_list[a_idx][1],unaltered_gain_array[a_idx],\
                                      unaltered_variance_array[a_idx],unaltered_basescore_array[a_idx] ))
            del index_value_list[a_idx]
            del UNSCALED_index_value_list[a_idx]
            del unaltered_gain_array[a_idx]
            del unaltered_meanPref_array[a_idx]
            del unaltered_variance_array[a_idx]
            del unaltered_basescore_array[a_idx]
            self.seen_features.update(self.plan_dataset[a[0]])
            if include_discovery_term_product:
                del discovery_values[a_idx]
                #discovery value is thought of as follows. IF there are two features of marginal probability 0.1, 0.15, then value of discov = 0.25
                #it is the upper bound of coverage of plans in which a feature might appear in. BIAS TO MORE DISCOVERY. could also use expected value, but not done
                discovery_values = [
                    1 + self.compute_discovery_value(self.plan_dataset[index_value_list[x][0]].difference(self.seen_features))
                    for x in range(len(index_value_list))]
            #now update the scores with the new discovery score.
        #end while

        #todo NOTE now we add one plan that is JUST about the discovery value of it
        if include_discovery_term_product:
            max_discov_value = max(discovery_values)
            if not max_discov_value == 1.0: #i.e. there are still features to explore
                max_discov_value_idx = discovery_values.index(max_discov_value)
                plan_idx_and_value = index_value_list[max_discov_value_idx]
                print("ADDING ONE PLAN PURELY for discovery")
                print("Discov plan", self.plan_dataset[plan_idx_and_value[0]])
                print("Discovery plan score is =", max_discov_value-1) #we used to add 1 because we multiply the BASE score by this value. So 1+x would scale it more
                chosen_indices.append(plan_idx_and_value[0])
                chosen_scores.append(plan_idx_and_value[1])
                self.seen_features.update(self.plan_dataset[plan_idx_and_value[0]])
            else: #DO NOTHING, DO NOT ADD ANOTHER RANDOM PLAN. throws off the comparison
                print("NO MORE FEATURES TO DISCOVER, NOT ADDING DISCOVERY PLANS")
                pass

        print("update_min max",self.min_rating,self.max_rating)
        print("TEMP PRINT chosen norm_E[gain]*norm_var values (with diversity) = ",chosen_scores)
        print("Overall statistics for CHOSEN norm_E[gain]*norm_var are ", summ_stats_fnc(chosen_scores))
        print("Overall statistics for ALL norm_E[gain]*norm_var are ", summ_stats_fnc(base_score+addendum))
        print("ROUND chosen plan stats =",chosen_plan_stats)
        self.indices_used.update(chosen_indices)
        return [self.plan_dataset[x] for x in chosen_indices]


    # ================================================================================================
    def store_annot_test_set(self,annot_test_set):
        """
        :return:
        """

        filtered_plans = []
        for plan in annot_test_set:
            if plan[0] != None:
                plan[2] = ["".join(x) for x in plan[2]]
                plan[3] = ["".join(x) for x in plan[3]]
                plan[4] = float(plan[4])
                filtered_plans.append(plan)
        # end for loop
        #we do NOT update the indices, if there are new features these new features are unused for predictions
        self.test_set = filtered_plans

    # ================================================================================================
    def get_next_round_plans_for_intf(self):
        """

        :return:
        """

        next_round_plans = self.plans_for_subseq_rounds[0]
        self.plans_for_subseq_rounds = self.plans_for_subseq_rounds[1:]
        #additionally for each plan, insert the previously annotated features (separate for liked and disliked)
        # todo NOTE cannot just check the features in the formatted dict because rare features below cutoff will not be there
        next_round_plans = insert_liked_disliked_features(next_round_plans,self.liked_features,self.disliked_features)
        print(next_round_plans)
        return next_round_plans

    # ================================================================================================
    def prep_next_round_plans_for_intf(self):
        """
        :return:
        """

        #todo NOTE THIS FUNCTION is called by the subprocess function that handles the manager
        print("PREPPING plans for round ", self.curr_round)
        chosen_plans = []
        if self.curr_round > self.num_rounds:
            #choose the best and worst plans based on the model learned so far
            self.test_set = self.select_best_and_worst(self.plans_per_round, quick_select=False,
                                                       num_check_to_select=5000)
            print("SENDING TEST SET PLANS printout from main manager = ",self.test_set)
            chosen_plans = self.test_set
        elif self.curr_round == 0:
            chosen_plans = self.sample_by_DBS(self.plans_per_round)
        elif self.curr_round % 2 != 0:
            print("prepped DBS half of plans")
            chosen_plans = self.sample_by_DBS(int(self.plans_per_round * Manager.RATIO_PLANS_EXPLORATION))
        else: #it is an RBUS round = even rounds after 0 (0 case is handled by the first if)
            print("prepped RBUS half of plans")
            # because integer always round down
            chosen_plans = self.sample_by_RBUS(int(self.plans_per_round * (1- Manager.RATIO_PLANS_EXPLORATION))+1)
        #---now package the plans in a string format that can be consumed by the web interface
        self.curr_round += 1
        self.plans_for_subseq_rounds.append([string_char_split(x[2]) for x in chosen_plans]) #the 2nd index is for the feature string. split into list of chars

    # ================================================================================================
    def check_next_round_RBUS(self):
        """
        :return:
        """

        return self.curr_round % 2 == 0 and self.curr_round != 0 and self.curr_round <= self.num_rounds


    # ================================================================================================
    def encode_by_relevant_features(self,formatted_plan_idx):
        """

        :param formatted_plan_idx:
        :return:
        """

        plan_features = self.plan_dataset[formatted_plan_idx][1]
        plan_encoding = np.zeros(self.CONFIRMED_features_dimension, dtype=float)
        for single_feature in plan_features:
            if single_feature in self.CONFIRMED_features:
                plan_encoding[self.RBUS_indexing.index(single_feature)] = 1
        return plan_encoding

    # ================================================================================================
    @staticmethod
    def check_for_feature(feature_seq_tuple, all_features):
        """
        :param num_test_plans:
        :return:
        """
        return feature_seq_tuple in all_features



    # ================================================================================================

    def extract_test_set(self,test_set_size):
        """

        :param num_test_plans:
        :return:
        """
        random.seed(self.random_seed)
        test_set_indices = random.sample(range(len(self.plan_dataset)),test_set_size)
        #remove indices from available indices
        self.indices_used.update(test_set_indices)
        return [self.plan_dataset[x] for x in test_set_indices]




    # ================================================================================================
    #todo redo this by simply making a function for "compute_plan_feature_distance"
    # def sample_randomly_wDiversity(self, num_samples):
    #     """
    #     Take plan if greater than avg feature distance over all other seen plans.
    #     :param num_samples:
    #     :return:
    #     """
    #     chosen_plans = []
    #     available_indices = set(range(len(self.plan_dataset)))
    #     available_indices.difference_update(self.indices_used)
    #     available_indices.difference(self.tried_indices)
    #     while len(chosen_plans) < num_samples:
    #         curr_index = random.sample(available_indices,1)[0]
    #         curr_plan = self.plan_dataset[curr_index]
    #         avg_feat_distance = self.average_feat_distance #this is to handle the case when no seen indices are there
    #         for single_seen_index in self.indices_used:
    #             avg_feat_distance += \
    #                 self.compute_plan_feature_distance(curr_plan,self.plan_dataset[single_seen_index])
    #         #end for loop
    #         avg_feat_distance /= len(self.indices_used)+1 #+1 is for the starting point of "avg_feat_distance"
    #         if avg_feat_distance >= self.average_feat_distance:
    #             chosen_plans.append(curr_plan)
    #             self.indices_used.add(curr_index)
    #         else:
    #             self.tried_indices.add(curr_index)
    #         #dont repeat this index
    #         available_indices.remove(curr_index)
    #     #end while loop
    #     return chosen_plans




    # ================================================================================================
    def sample_randomly(self, num_samples):
        """
        Sample Plans randomly to test for oracle feedback and R-BUS strategy
        sampled plans are considered used
        :param num_samples:
        :return:
        """

        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        available_indices.difference(self.tried_indices)
        new_indices_sampled = random.choices(list(available_indices), k=num_samples)
        for idx in new_indices_sampled:
            self.seen_features.update(self.plan_dataset[idx])
        self.indices_used.update(new_indices_sampled)
        return [self.plan_dataset[x] for x in new_indices_sampled]


    # ================================================================================================
    @staticmethod
    def unknown_feature_score_computation(input_list):
        """

        :param input_list:
        :return:
        """
        def get_gain_function(min_value, max_value, lower_percentile = 0.2, upper_percentile = 0.8):
            """
            :summary: abstract gain function
            :param x_value:
            :return:
            :return:
            """
            range = max_value - min_value
            lower_fifth = min_value + range * lower_percentile
            upper_fifth = min_value + range * upper_percentile
            scaled_std_dev = range * 0.1
            return get_biModal_gaussian_gain_function(lower_fifth, upper_fifth, scaled_std_dev)
        # --------------
        def get_biModal_gaussian_gain_function(a=0.2, b=0.8, sd=0.1):
            """
            :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
            There is no special property with probabilities and summing to 1 in this application. Only relative gain values
            that match what we need.
            :param x_value:
            :param a: IS NOT PERCENTILE, but absolute values
            :param b:
            :return: the function
            """
            gain_a = norm(a, sd)
            gain_b = norm(b, sd)
            return lambda x: (gain_a.pdf(x) + gain_b.pdf(x)) / 2
        #-----end inner function get_biModal_gaussian_gain_function

        min_rating, max_rating, include_gain  = input_list
        preference_possible_values = np.linspace(min_rating, max_rating, num=NUM_SAMPLES_XAXIS_SAMPLES)
        gain_function = get_gain_function(min_rating,max_rating)
        #todo index 3 the distribution for unknown feature weights is uniform
        preference_prob_density = [1/(max_rating-min_rating)]*len(preference_possible_values)
        normalizing_denom = np.sum(preference_prob_density)
        if normalizing_denom == 0.0:
            normalizing_denom = 1.0 # THIS IS VERY HACKY and needed to avoid nan, that occurs when there are no features, and alpha is deterministic
        #todo NOTE this is HACKY/SAMPLING APPROACH of computing the mean value. It is not mathematically valid, but WORKS if x axis is sampled sufficiently
        # AND very fast + works well
        mean_preference = np.sum(preference_prob_density * preference_possible_values) / normalizing_denom
        preference_variance = np.sum(np.square(preference_possible_values - mean_preference) * preference_prob_density) / normalizing_denom
        composite_func_integral = 0.0
        if include_gain:  # saves time when we do not use gain function
            gain_outputs = np.array([gain_function(x) for x in preference_possible_values])
            composite_func_outputs = preference_prob_density * gain_outputs #elementwise product of two vectors
            composite_func_outputs = composite_func_outputs.reshape(composite_func_outputs.shape[0])
            #todo MAYBE USE THE SAMPLING APPROACH HERE AS WELL !! speed it up considerably !!
            composite_func_integral = integrate.trapz(composite_func_outputs, preference_possible_values)
        # print("TEMP PRINT: The composite_func_integral of the prob*gain over the outputs is =", composite_func_integral)
        # this composite_func_integral is the Expected gain from taking this sample
        return composite_func_integral, preference_variance


    # ================================================================================================
    @staticmethod
    def parallel_variance_computation(input_list):
        """

        :param input_list:
        :return:
        """
        def get_gain_function(min_value, max_value, lower_percentile = 0.2, upper_percentile = 0.8):
            """
            :summary: abstract gain function
            :param x_value:
            :return:
            :return:
            """
            range = max_value - min_value
            # lower_fifth = min_value + range * lower_percentile
            # upper_fifth = min_value + range * upper_percentile
            scaled_std_dev = range * 0.1 #20% of the extreme ratings. 10% on either side
            return get_biModal_gaussian_gain_function(min_value, max_value, scaled_std_dev)
        # --------------
        def get_biModal_gaussian_gain_function(a=0.2, b=0.8, sd=0.1):
            """
            :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
            There is no special property with probabilities and summing to 1 in this application. Only relative gain values
            that match what we need.
            :param x_value:
            :param a: IS NOT PERCENTILE, but absolute values
            :param b:
            :return: the function
            """
            gain_a = norm(a, sd)
            gain_b = norm(b, sd)
            return lambda x: (gain_a.pdf(x) + gain_b.pdf(x)) / 2
        #-----end inner function get_biModal_gaussian_gain_function

        learning_model_bayes,encoded_plan, min_rating, max_rating, include_gain  = input_list
        try:
            predictions = learning_model_bayes.get_outputs_from_distribution(encoded_plan, num_samples=NUM_SAMPLES_KDE)
        except:
            return 1.0 , 1.0,1.0 #when no features have been discovered yet
        mean_preference = np.mean(predictions)
        preference_variance = np.sum(np.square(predictions - mean_preference) )/(len(predictions)-1)


        mean_gain_value = 0.0
        if include_gain:  # saves time when we do not use gain function
            gain_function = get_gain_function(min_rating, max_rating)
            gain_values = np.array([gain_function(x) for x in predictions])
            # todo NOTE this is HACKY/SAMPLING APPROACH of computing the mean value. It is not mathematically valid, but WORKS if well sampled and "uniform".
            # AND very fast + works well
            mean_gain_value = np.mean(gain_values)
        # print("TEMP PRINT: The composite_func_integral of the prob*gain over the outputs is =", composite_func_integral)
        # this composite_func_integral is the Expected gain from taking this sample
        return mean_gain_value, preference_variance, mean_preference


    # ================================================================================================
    def set_num_cores_RBUS(self,num_cores):
        """
        """
        self.num_cores_RBUS = num_cores


    # ================================================================================================
    def reformat_features_and_update_indices(self, annotated_plans):
        """

        :param annotated_plans:
        :return:
        """
        filtered_plans = []
        for plan in annotated_plans:
            if plan[0] != None:
                plan[2] = [tuple(x) for x in plan[2]]
                plan[3] = [tuple(x) for x in plan[3]]
                try:
                    plan[4] = float(plan[4]) #handles empty string and incorrect format string.
                except ValueError:
                    plan[4] = float(0.0)
                filtered_plans.append(plan)
        #end for loop
        self.update_indices(filtered_plans)

    # ================================================================================================



    def update_indices(self, annotated_plans):
        """
        Takes annotated plans and update indices required for R-BUS and DBS modules
        :param annotated_plans: format - [[state-sequence],
        [list of features], [liked features], [disliked features], rating]
        :return:
        """
        self.annotated_plans_by_round.append(annotated_plans)
        liked_features = set()
        disliked_features = set()
        irrelev_features = set()
        for single_plan in annotated_plans:
            liked_features.update(single_plan[1])
            disliked_features.update(single_plan[2])
            irrelev_features.update(single_plan[0].difference(liked_features.union(disliked_features)))
            if len(single_plan[1]) + len(single_plan[2]) == 0:
                continue #disregard this plan
            self.annotated_plans.append(single_plan)
        #end for loop through annotated plans

        #IMPORTANT remove plan duplicates THAT COULD OCCUR because irrelevant features dont count.
        #IT WOULD BE THE SAME DATAPOINT, and thus reduce the number of effective samples
        temp_plan_dataset = set()
        for plan_idx in range(len(self.plan_dataset)):
            #filter even if a duplicate with a plan in the used indices
            plan = self.plan_dataset[plan_idx]
            temp_plan = tuple(set(plan).difference(self.IRrelevant_features))
            if temp_plan not in temp_plan_dataset:
                temp_plan_dataset.add(temp_plan)
            else: #this is a duplicate in terms of the not-irrelevant features
                self.indices_used.add(plan_idx) #remove it from consideration
            #end outer if
        #end for

        #end for

        #NOTE this is incase there were some discrepancies or errors in the annotation.
        #todo INDEX 4: This will turn off learning about liked, disliked, and irrelevant features
        if self.use_feature_feedback:
            irrelev_features = irrelev_features.difference(liked_features.union(disliked_features))
            self.liked_features.update(liked_features)
            self.disliked_features.update(disliked_features)
            self.IRrelevant_features.update(irrelev_features)
            self.CONFIRMED_features = self.liked_features.union(self.disliked_features)
            #Now reindex the features after the removals
            self.POSSIBLE_features = self.all_s1_features.difference(self.IRrelevant_features)
            self.CONFIRMED_features_dimension = len(self.CONFIRMED_features)
            # self.RBUS_indexing = sorted(list(self.POSSIBLE_features))
            # self.RBUS_prior_weights = np.zeros(self.CONFIRMED_features_dimension)

            prev_round_rbus_indexing = self.RBUS_indexing
            prev_round_rbus_prior_mean = self.RBUS_prior_mean
            prev_round_rbus_prior_var = self.RBUS_prior_var
            self.RBUS_indexing = sorted(list(self.CONFIRMED_features))
            self.RBUS_prior_mean = np.zeros(len(self.CONFIRMED_features))
            self.RBUS_prior_var = np.zeros(len(self.CONFIRMED_features))


            for single_feature in self.CONFIRMED_features:
                if single_feature in self.liked_features:
                    prior_mean = self.like_dislike_prior_mean[0]
                    prior_var = self.like_dislike_prior_var[0]
                    try:
                        prior_mean = prev_round_rbus_prior_mean[prev_round_rbus_indexing.index(single_feature)]
                        prior_var = prev_round_rbus_prior_var[prev_round_rbus_indexing.index(single_feature)]
                    except:
                        pass
                    self.RBUS_prior_mean[self.RBUS_indexing.index(single_feature)] = prior_mean
                    self.RBUS_prior_var[self.RBUS_indexing.index(single_feature)] = prior_var

                else:
                    prior_mean = self.like_dislike_prior_mean[1]
                    prior_var = self.like_dislike_prior_var[0]
                    try:
                        prior_mean = prev_round_rbus_prior_var[prev_round_rbus_indexing.index(single_feature)]
                        prior_var = prev_round_rbus_prior_var[prev_round_rbus_indexing.index(single_feature)]
                    except:
                        pass
                    self.RBUS_prior_mean[self.RBUS_indexing.index(single_feature)] = prior_mean
                    self.RBUS_prior_var[self.RBUS_indexing.index(single_feature)] = prior_var
        #end if self.use_feature_feedback:

    #================================================================================================
    def get_feedback(self,all_plans):
        """

        :param plans:
        :return:
        """
        annot_plans = []
        #todo SAVE all the feedback annotations and ratings, so the noise effect is the same when testing the different methods
        if len(all_plans) > 0:
            print("ANNOTATION NEW PLANS of size ",len(all_plans))
        newly_annot_plans = self.sim_human.get_feedback(all_plans)
        annot_plans += newly_annot_plans
        sorted_annot_plans = sorted(annot_plans, key = lambda x:x[-1])
        if len(sorted_annot_plans)>100:
            print("ONLY SOME !! of the feedback given was =", random.sample(sorted_annot_plans,100))

        return annot_plans
    #================================================================================================
    def store_all_plans_feedback(self):
        """
        :return:
        """
        remaining_plans = [self.plan_dataset[x] for x in range(len(self.plan_dataset)) if x not in self.indices_used]
        self.get_feedback(remaining_plans)




    # ================================================================================================
    def relearn_model(self, learn_LSfit = False, num_chains=1):
        """
        since we have the relevant features and some annotated plans(<plans, rating>, we learn a liner regression model
        by Bayesian Learning. The manager will connect to the learning engine to learn and update the model
        :return:
        """
        if len(self.annotated_plans) == 0:
            return
        rescaled_plans = copy.deepcopy(self.annotated_plans)
        if len(self.annotated_plans) == 0: #this can happen when the user has not rated anything and just clicked
            #create a dummy plan with no features and preference = 0
            rescaled_plans = [[{},[],[],0.0]]

        #todo NOTe rescaling is currently removed but we still need to track min and max rating

        ratings = np.array([x[-1] for x in rescaled_plans])
        min_rating = np.min(ratings)
        max_rating = np.max(ratings)
        if min_rating < self.min_rating:
            self.min_rating = min_rating
        if min_rating < self.annotated_min:
            self.annotated_min = min_rating
        #---
        if max_rating > self.max_rating:
            self.max_rating = max_rating
        if max_rating > self.annotated_max:
            self.annotated_max = max_rating
        scores = []
        for plan in self.annotated_plans:
            scores.append(plan[-1])
        encoded_plans_list = []
        for single_plan in rescaled_plans:
            encoded_plan = [np.zeros(self.CONFIRMED_features_dimension), single_plan[3]]

            for single_feature in single_plan[1] + single_plan[2]: #the liked and disliked features
                encoded_plan[0][self.RBUS_indexing.index(single_feature)] = 1
            encoded_plans_list.append(encoded_plan)

        MLE_reg_model = None
        # if learn_LSfit:
        if True:
            from sklearn import linear_model
            # MLE_reg_model = linear_model.LinearRegression(fit_intercept=True) #NORMALIZE wont help, the input is binary. Already normalized
            # MLE_reg_model = linear_model.LinearRegression(fit_intercept=False) #NORMALIZE wont help, the input is binary. Already normalized
            MLE_reg_model = linear_model.Ridge(fit_intercept=False) #normalize wont help here either, the input is binary, already normalized
            input_dataset = np.array([x[0] for x in encoded_plans_list])
            output_dataset = np.array([x[1] for x in encoded_plans_list])

            #todo PLEASE test the weighted fit more on toy problems. When you normalized the weights, it failed miserably
            # which was unexpected
            # WEIGHTS ARE CURRENTLY TURNED OFF, VERY ERRATIC PERFORMANCE
            # weight_func = RBUS_selection.get_gain_function(self.min_rating,self.max_rating)
            # weights = [weight_func(x) for x in output_dataset]
            # weights = [1.0 for x in output_dataset]

            MLE_reg_model.fit(input_dataset, output_dataset)
            print("Coefficients's values ", MLE_reg_model.coef_)
            print("Intercept: %.4f" % MLE_reg_model.intercept_)
            self.model_MLE = MLE_reg_model
        #end if learn_LSfit

        print("RELEARINING BAYESIAN MODEL")
        prior_dict = {}
        for single_feature in self.CONFIRMED_features:
            prior_dict[single_feature] = (self.RBUS_prior_mean[self.RBUS_indexing.index(single_feature)],self.RBUS_prior_var[self.RBUS_indexing.index(single_feature)] )
        print("priors are = ", prior_dict)
        print("NOTE that the noise is ALWAYS set to EXPECTED_NOISE_VARIANCE = 1.0, which is bad.")

        self.learning_model_bayes.learn_bayesian_linear_model(encoded_plans_list,
                                                              np.array(self.RBUS_prior_mean),
                                                              self.CONFIRMED_features_dimension,
                                                              sd= EXPECTED_NOISE_STD_DEV,
                                                              sampling_count=2000,
                                                              num_chains=num_chains,
                                                              uninformative_prior_sd = np.diag(self.RBUS_prior_var))
                                                              # num_chains=num_chains)

        # ALSO UPDATE THE PRIORS FOR THE NEXT ROUND. WE START OPTIMISITC AND UPDATE THEM TOWARDS THEIR TRUE VALUES
        param_stats = [self.learning_model_bayes.linear_params_values["betas"][0:2000, x] for x in
                       range(self.CONFIRMED_features_dimension)]
        param_means = np.array([np.mean(x) for x in param_stats])
        param_var = [np.var(x) for x in param_stats]
        self.RBUS_prior_mean = np.zeros(len(self.CONFIRMED_features))
        self.RBUS_prior_var = np.zeros(len(self.CONFIRMED_features))

        for single_feature in self.CONFIRMED_features:
            if single_feature in self.liked_features:
                self.RBUS_prior_mean[self.RBUS_indexing.index(single_feature)] = \
                    param_means[self.RBUS_indexing.index(single_feature)]
                self.RBUS_prior_var[self.RBUS_indexing.index(single_feature)] = \
                    param_var[self.RBUS_indexing.index(single_feature)]
                # todo note, this was the other option, to agressively update the prior towards zero by mean-std_dev
                # instead we assume just the mean, as it is a more conservative update. Aggressive update maybe good too. never tested
                # param_abs[self.RBUS_indexing.index(single_feature)] - math.sqrt(
                #     param_vars[self.RBUS_indexing.index(single_feature)])
            else:
                self.RBUS_prior_mean[self.RBUS_indexing.index(single_feature)] = \
                    param_means[self.RBUS_indexing.index(single_feature)]
                self.RBUS_prior_var[self.RBUS_indexing.index(single_feature)] = \
                    param_var[self.RBUS_indexing.index(single_feature)]
                # todo note, this was the other option, to agressively update the prior towards zero by mean-std_dev
                # -1*param_abs[self.RBUS_indexing.index(single_feature)] - math.sqrt(
                #     param_vars[self.RBUS_indexing.index(single_feature)])
            # end else
        #end for loop through confirmed features

        #TODO Update the noise estimate.
        # take the variance of the error as the noise.  YES ! we just assume that we are right. Initially, this would be large, and then
        # as we get better parameter estimates, we (in tandem) get a better estimate of the noise

        if self.sim_human != None:  #i.e. we are in simulated testing
            # continue from here to update params
            #end for updating confirmed features.
            param_summ_stats = [summ_stats_fnc(x) for x in param_stats]
            bayes_feature_dict = copy.deepcopy(self.sim_human.feature_preferences_dict)
            for single_feature in bayes_feature_dict:
                try:
                    bayes_feature_dict[single_feature].append(param_summ_stats[self.RBUS_indexing.index(single_feature)])
                except ValueError: # for the feature not being in the list
                    pass

            if MLE_reg_model != None:
                MLE_feature_dict = copy.deepcopy(self.sim_human.feature_preferences_dict)
                for single_feature in MLE_feature_dict:
                    try:
                        MLE_feature_dict[single_feature].append(MLE_reg_model.coef_[self.RBUS_indexing.index(single_feature)])
                    except ValueError:  # for the feature not being in the list
                        pass


            #todo not just mean, do MODE of features
            print("RATINGS seen are = ", summ_stats_fnc(ratings))
            print("Bayes params ","  ||  ".join([str(x) for x in bayes_feature_dict.items()]))
            print("Bayes intercept summ stats")
            print(summ_stats_fnc(self.learning_model_bayes.linear_params_values["alpha"][0:2000]))
            if MLE_reg_model != None:
                print("MLE params ","  ||  ".join([str(x) for x in MLE_feature_dict.items()]))
            # self.evaluate(self.test_set)

    #================================================================================================
    def evaluate(self,annotated_test_plans=None):
        """
        :summary : we have the actual result in the annotated test plans, we see what the model
        would have predicted , which is a distribution over values. We take the mode of the output distribution
        as the prediction. Then the error is RMS error over the test set

        :param annotated_test_plans:
        :return:
        """
        if annotated_test_plans == None:
            annotated_test_plans = self.test_set

        #TODO NOTE WE FILTER THE PLANS WITH NO FEATURES ARE RATED 0.0 (not interesting) ONLY FOR TESTING, NOT FOR TRAINING (UNKNOWN THEN)
        annotated_test_plans = [x for x in annotated_test_plans if x[-1] != 0.0]

        MLE_total_squared_error = 0.0
        MLE_error_list = []
        bayes_total_error = 0.0
        bayes_error_list = []
        bayes_target_prediction_list = []
        MLE_final_error = 0.0
        MLE_target_prediction_list = []
        bayes_final_error = 0.0

        count_samples = 0


        #end if
        print(" IN EVALUATION ",self.curr_round ,self.num_rounds+1)
        if self.curr_round > self.num_rounds+1: #then all rounds were completed
            self.completed = True

        print("=====EVALUATION===== after round ",self.curr_round/2-1)#recall we double the round numbers to handle dbs and rbus separation, except for first round which is only dbs.
        for single_annot_plan_struct in annotated_test_plans:

            current_plan_features = single_annot_plan_struct[1] + single_annot_plan_struct[2]
            encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
            count_samples +=1
            for single_feature in current_plan_features:
                if single_feature in self.CONFIRMED_features:
                    encoded_plan[self.RBUS_indexing.index(single_feature)] = 1

            true_value = float(single_annot_plan_struct[-1])
            #end for loop
            if self.model_MLE != None:
                mle_predict = self.model_MLE.predict([encoded_plan])[0]
                current_abs_error = abs(true_value - mle_predict)
                MLE_total_squared_error += current_abs_error
                MLE_error_list.append(current_abs_error)
                MLE_target_prediction_list.append((true_value,mle_predict))
            else:
                MLE_total_squared_error += 0.0
                MLE_error_list.append(0.0)
                MLE_target_prediction_list.append((true_value,0.0))

            try:
                predictions = self.learning_model_bayes.get_outputs_from_distribution(encoded_plan, num_samples=NUM_SAMPLES_KDE)
                mean_prediction = np.mean(predictions)
                prediction_variance = np.var(predictions)
            except:
                mean_prediction = 0.0
                prediction_variance = 0.0

            # preference_possible_values = np.linspace(self.min_rating, self.max_rating, num=NUM_SAMPLES_XAXIS_SAMPLES)
            # preference_prob_density = kernel(preference_possible_values)
            # if not np.min(preference_prob_density) == np.max(preference_prob_density):
            #     index_mode= np.where(preference_prob_density == np.max(preference_prob_density))[0][0]
            #     mode_prediction = preference_possible_values[index_mode]
            #     normalizing_denom = np.sum(preference_prob_density)
            #     mean_prediction = np.sum(preference_prob_density * preference_possible_values)/normalizing_denom #this is a HACK but computationally faster
            #     prediction_variance = np.sum(np.square(preference_possible_values-mean_prediction)*preference_prob_density)/normalizing_denom #this is a HACK, not true variance, but computationally faster
            # else:
            #     #absolute uniform distribution over all preference values, actually means we have no information,
            #     # we SET the alpha to be fixed, so no variations in the output.
            #     mode_prediction = 0.0
            #     mean_prediction = 0.0
            #     prediction_variance = 0.0

            #todo NOTE USING MEAN PREDICTION, makes more sense with using variance for decisions
            current_abs_error = abs(true_value - mean_prediction)
            bayes_total_error += current_abs_error
            bayes_error_list.append( current_abs_error)
            bayes_target_prediction_list.append((true_value, mean_prediction, prediction_variance))


        if count_samples == 0:
            print("NOT TEST PLANS WERE RATED FAILED")
            return bayes_final_error,MLE_error_list

        #end for loop
        if self.model_MLE != None:
            MLE_final_error = MLE_total_squared_error / count_samples
            print("LINEAR MODEL The average error in ALL regions is = ", MLE_final_error)
            print("LINEAR MODEL Error Statistics of ALL regions, ", summ_stats_fnc(MLE_error_list))
            # print("LINEAR MODEL target and prediction ", MLE_target_prediction_list
        else:
            MLE_final_error = 0.0

        #end if
        bayes_final_error = bayes_total_error / count_samples
        print("BAYES MODEL The average error in ALL regions= ", bayes_final_error)
        print("BAYES MODEL Error Statistics of ALL regions , ",summ_stats_fnc(bayes_error_list))
        # print("BAYES MODEL target and prediction ",bayes_target_prediction_list)
        if self.results_by_round == None:
            self.results_by_round = []
        self.results_by_round = self.results_by_round.append([bayes_final_error,MLE_final_error])
        # print("bayes error list ", bayes_error_list)
        # print("MLE error list ", MLE_error_list)

        return bayes_final_error,MLE_final_error

    # ================================================================================================

    def get_balanced_test_set_before_training(self, test_set_size=1000, ratio_interesting_region=0.4):
        """
        :summary : extract plans such that we have a portion of the plans in the interesting region
        :param test_set_size:
        :param ratio_interesting_region:
        :return:
        plans are selected like
           ||xxxxx.....******|*******.....xxxxx|| where x is interesting region, * is just above and below the middle, and "." is in between
        """

        # in the following line we div by 2 since we want plans with ratings at either extreme
        per_region_num_interesting_plans = int(test_set_size * ratio_interesting_region / 2)
        num_trivial_plans = test_set_size - 2 * per_region_num_interesting_plans
        # the above trivial plans are split into two groups as well just below the preferred mark, and just above the rejected mark as in the function comments
        # TODO remove all the print statements and integral checks, will speed things up considerably
        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        # print("num available points = ",len(available_indices))
        index_value_list = []  # a list of tuples of type (index,value)
        with CodeTimer():
            # compute and order the predicted scores for the remaining plans
            self.sim_human.change_rating_noise(0.0) #the balanced dataset depends on true ratings
            rated_annot_plans = self.sim_human.get_feedback(self.plan_dataset)
            rated_annot_plans = enumerate(rated_annot_plans,0) #start from zero
            sorted_all_results = sorted(rated_annot_plans, key=lambda x: x[1][-1], reverse=True)
            num_results = len(sorted_all_results)
            # get plans from top, bottom, and middle
            print("******EXPECTED RATINGS in interesting regions*********")
            print(sorted_all_results[0:per_region_num_interesting_plans] + sorted_all_results[
                                                                           -per_region_num_interesting_plans:])
            # todo ADD diversity to the samples chosen. Avoid the same RELEVANT feature encoding. May not be enough then though ??
            chosen_indices = [x[0] for x in
                              sorted_all_results[0:per_region_num_interesting_plans]]  # most preferred value
            chosen_indices += [x[0] for x in
                               sorted_all_results[int(num_results / 2 - num_trivial_plans / 2):int(
                                   num_results / 2)]]  # just above median
            chosen_indices += [x[0] for x in
                               sorted_all_results[-per_region_num_interesting_plans:]]  # most hated/least preferred
            chosen_indices += [x[0] for x in
                               sorted_all_results[int(num_results / 2):int(
                                   num_results / 2 + num_trivial_plans / 2)]]  # just below median
            # sort the results are get plans on either end
        self.indices_used.update(chosen_indices)
        return [self.plan_dataset[x] for x in chosen_indices]

    # ================================================================================================

    def get_extremities_test_set_before_training(self, test_set_size=1000,
                                                 lower_bound_cutoff_ratio = 0.1, upper_bound_cutoff_ratio = 0.9):
        """
        :summary : extract plans such that we have a portion of the plans in the interesting region
        :param test_set_size:
        :param percentile_interesting_region: if 0.2, then top 20% and bottom 20%
        :return:
        plans are selected like
           ||xxxxx.....******|*******.....xxxxx|| where x is interesting region, * is just above and below the middle, and "." is in between
        """

        # select every OTHER data point, not all the top and bottom ones. Leave some for training :-)

        # in the following line we div by 2 since we want plans with ratings at either extreme
        per_region_num_interesting_plans = int(test_set_size/2)
        # the above trivial plans are split into two groups as well just below the preferred mark, and just above the rejected mark as in the function comments
        # TODO remove all the print statements and integral checks, will speed things up considerably
        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        # print("num available points = ",len(available_indices))
        index_value_list = []  # a list of tuples of type (index,value)
        old_noise = self.sim_human.gaussian_noise_sd
        with CodeTimer():
            # compute and order the predicted scores for the remaining plans
            self.sim_human.change_rating_noise(0.0)  # the balanced dataset depends on true ratings
            rated_annot_plans = self.sim_human.get_feedback(self.plan_dataset)
            rated_annot_plans = enumerate(rated_annot_plans, 0)  # start from zero
            sorted_all_results = sorted(rated_annot_plans, key=lambda x: x[1][-1])
            min_rating = sorted_all_results[0][1][-1]
            max_rating = sorted_all_results[-1][1][-1]
            ratings_range = max_rating - min_rating


            num_results = len(sorted_all_results)
            # get plans from top, bottom, and middle
            # print("******EXPECTED RATINGS in interesting regions (we sample every other point)*********")
            # print(sorted_all_results[0:2*per_region_num_interesting_plans] + sorted_all_results[
            #                                                                -2*per_region_num_interesting_plans:])
            #now dont take all the top and bottom plans, alternate and sample
            chosen_indices = [x[0] for x in
                              sorted_all_results[1:per_region_num_interesting_plans:2]]  # most preferred value
            chosen_indices += [x[0] for x in
                               sorted_all_results[-per_region_num_interesting_plans::2]]  # most hated/least preferred
            # sort the results are get plans on either end
        self.indices_used.update(chosen_indices)
        self.sim_human.change_rating_noise(old_noise)
        return [self.plan_dataset[x] for x in chosen_indices]

    # ================================================================================================

    def get_balanced_test_set_after_training(self, test_set_size = 1000, ratio_interesting_region = 0.4):
        """
        :summary : extract plans such that we have a portion of the plans in the interesting region
        :param test_set_size:
        :param ratio_interesting_region:
        :return:
        plans are selected like
           ||xxxxx.....******|*******.....xxxxx|| where x is interesting region, * is just above and below the middle, and "." is in between
        """


        # in the following line we div by 2 since we want plans with ratings at either extreme
        per_region_num_interesting_plans = int(test_set_size*ratio_interesting_region/2)
        num_trivial_plans = test_set_size - 2*per_region_num_interesting_plans
        #the above trivial plans are split into two groups as well just below the preferred mark, and just above the rejected mark as in the function comments
        #TODO remove all the print statements and integral checks, will speed things up considerably
        available_indices = set(range(len(self.plan_dataset)))
        available_indices.difference_update(self.indices_used)
        # print("num available points = ",len(available_indices))
        index_value_list = [] # a list of tuples of type (index,value)
        with CodeTimer():
            #compute and order the predicted scores for the remaining plans
            all_results = []
            for single_plan_idx in available_indices:
                current_plan = self.plan_dataset[single_plan_idx][1]
                encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
                for single_feature in current_plan:
                    if single_feature in self.CONFIRMED_features:
                        encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
                #end for loop through current plan
                #the last false is for including gain, we do not care about that for output prediction

                try:
                    predictions = self.learning_model_bayes.get_outputs_from_distribution(encoded_plan,
                                                                                          num_samples=NUM_SAMPLES_KDE)
                    mean_prediction = np.mean(predictions)
                except:
                    mean_prediction = 0.0
                all_results.append((single_plan_idx,mean_prediction))
            #end for loop through the available indices
            sorted_all_results = sorted(all_results,key= lambda x:x[1],reverse=True)
            num_results = len(sorted_all_results)
            #get plans from top, bottom, and middle
            print("******EXPECTED RATINGS in interesting regions*********")
            print(sorted_all_results[0:per_region_num_interesting_plans] + sorted_all_results[-per_region_num_interesting_plans:])
            #todo ADD diversity to the samples chosen. Avoid the same RELEVANT feature encoding. May not be enough then though ??
            chosen_indices = [x[0] for x in sorted_all_results[0:per_region_num_interesting_plans]] #most preferred value
            chosen_indices += [x[0] for x in
                sorted_all_results[ int(num_results/2-num_trivial_plans/2):int(num_results/2)]] #just above median
            chosen_indices += [x[0] for x in sorted_all_results[-per_region_num_interesting_plans:]] #most hated/least preferred
            chosen_indices += [x[0] for x in
                sorted_all_results[int(num_results/2):int(num_results/2+num_trivial_plans/2)]] #just below median
            #sort the results are get plans on either end
        return [self.plan_dataset[x] for x in chosen_indices]


#=============================================================================
    def region_based_evaluation(self, annotated_test_plans, eval_output_regions=[0.2,0.8],
                                inside_region=True):
        """

        :param annotated_test_plans:
        :param eval_output_regions:
        :return:
        """
        test_plan_max = 0
        test_plan_min = 0
        for single_annot_plan_struct in annotated_test_plans:
            rating = single_annot_plan_struct[-1]
            if rating < test_plan_min:
                test_plan_min = rating
            elif rating > test_plan_max:
                test_plan_max = rating
        #end for loop



        true_values_and_diff = []
        #TODO NOTE WE FILTER THE PLANS WITH NO FEATURES ARE RATED 0.0 (not interesting) ONLY FOR TESTING, NOT FOR TRAINING (UNKNOWN THEN)
        annotated_test_plans = [x for x in annotated_test_plans if x[-1] != 0.0]
        bayes_total_error = 0.0
        UNALTERED_bayes_total_error = 0.0
        bayes_error_list = []
        MLE_total_error = 0.0
        bayes_target_prediction_list = []
        MLE_target_prediction_list = []
        MLE_error_list = []
        # convert the percentiles to actual regions
        sorted_ratings = sorted([x[-1] for x in annotated_test_plans])
        num_plans = len(annotated_test_plans)
        range_of_values = test_plan_max-test_plan_min
        cutoff_regions = [range_of_values*(eval_output_regions[0])+test_plan_min , range_of_values*(eval_output_regions[1])+test_plan_min]
        # for single_region in eval_percentile_regions:
        #     bound_indices = [int(x * num_plans) for x in single_region]
        #     if num_plans in bound_indices:
        #         bound_indices[bound_indices.index(num_plans)] = num_plans - 1
        #     cutoff_regions.append([sorted_ratings[x] for x in bound_indices])

        # ratings_range = self.max_rating - self.min_rating
        # cutoff_regions = [(self.min_rating + x[0]*ratings_range, self.min_rating + x[1]*ratings_range) for x in eval_percentile_regions]
        count_samples = 0
        error_scaler = 1 # MAKES NO DIFFERENCE, Just scales values, max(abs(test_plan_min),abs(test_plan_max))
        # print(
        #     "NOTE WE ASSUME A PLAN WITH NO KNOWN FEATURES IS OF VALUE 0, AND SO NOT COUNTED IN THE TEST SET EVALUATION")
        # print("ALL RATINGS are = ", sorted_ratings)
        print("Cutoffregions = ", cutoff_regions)
        print("TEST PLAN MIN MAX ",test_plan_min, test_plan_max)

        filtered_plans = [x for x in annotated_test_plans if not(cutoff_regions[0] < x[-1] and x[-1] < cutoff_regions[1])]

        for single_annot_plan_struct in filtered_plans:
            true_value = float(single_annot_plan_struct[-1])
            # if cutoff_regions[0] < true_value and true_value < cutoff_regions[1] : # we are inside the range, then not an extreme point
            #     continue
            count_samples += 1
            current_plan = single_annot_plan_struct[0]
            encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
            for single_feature in current_plan:
                if single_feature in self.CONFIRMED_features:
                    encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
            # ---first do the simple MLE model, easier

            if self.model_MLE != None:
                mle_predict = self.model_MLE.predict([encoded_plan])[0]
                current_abs_error = abs(
                    single_annot_plan_struct[-1] - mle_predict)
                MLE_total_error += current_abs_error
                MLE_error_list.append(current_abs_error)
                MLE_target_prediction_list.append((true_value,mle_predict))
            else:
                MLE_total_error += 0.0
                MLE_error_list.append(0.0)
                MLE_target_prediction_list.append((true_value,0.0))

            # ---now do the bayes model, need to get the MODE prediction
            try:
                predictions = self.learning_model_bayes.get_outputs_from_distribution(encoded_plan, num_samples=NUM_SAMPLES_KDE)
                mean_prediction = np.mean(predictions)
                prediction_variance = np.var(predictions)
            except:
                mean_prediction = 0.0
                prediction_variance = 0.0

            current_abs_error = abs(true_value - mean_prediction)
            current_abs_error = current_abs_error*abs(true_value)/error_scaler
            UNALTERED_error = abs(true_value - mean_prediction)
            bayes_total_error += current_abs_error
            UNALTERED_bayes_total_error += UNALTERED_error

            bayes_error_list.append( current_abs_error)
            bayes_target_prediction_list.append((true_value, mean_prediction, prediction_variance))
            true_values_and_diff.append((true_value, true_value - mean_prediction, prediction_variance))#the error with sign is passed on

        # end for loop
        print(" If inside INTERESTING REGION is ", inside_region)
        if count_samples == 0:
            print("THERE WERE ZERO SAMPLES ACCORDING TO OUR CURRENT RATING MAX!!")
            print("This can happen when the training set has a higher upper bound than the test set")
            # todo INSTEAD OF DOING TOP 20% BY MAX RANGE, should do upper 20% OF POINTS. BUT THIS DEFEATS THE PURPOSE, SO NO
            # WE AIM TO FIND TOP 20% AND BOTTOM 20% BY RANGE, not by points.
            return 0.0

        print("NUM SAMPLES = ", count_samples, " for when INTERESTING REGION IS", inside_region)
        if self.model_MLE != None:
            MLE_final_error = MLE_total_error / count_samples
            print("LINEAR MODEL The average REGION error is = ", MLE_final_error, "for percentile regions ",
                  eval_output_regions)
            print("LINEAR MODEL Error Statistics of CHOSEN regions , ", summ_stats_fnc(MLE_error_list))
            # print("LINEAR MODEL target and prediction ", MLE_target_prediction_list)
        else:
            MLE_final_error = 0.0

        # end if
        bayes_final_error = bayes_total_error / count_samples
        UNALTERED_bayes_final_error = UNALTERED_bayes_total_error / count_samples
        print("BAYES MODEL The average REGION error is = ", bayes_final_error, "for percentile regions ",
              eval_output_regions)
        print("BAYES MODEL Error Statistics of CHOSEN regions , ", summ_stats_fnc(bayes_error_list))
        # print("BAYES MODEL target and prediction ",bayes_target_prediction_list)

        return bayes_final_error,UNALTERED_bayes_final_error,MLE_final_error,true_values_and_diff

# ================================================================================================

    def select_best_and_worst(self, num_samples, quick_select = False, num_check_to_select = 5000):
        """
        :summary: select the highest rated and lowest rated plans.
        We include the variance in the rating. (x-mu) + (x-mu)*(1-NORM_var)
        :return:
        """
        #need to compute prediction for all remaining plans ... could take a while.
        #? if it takes too long, then try sampling a random set of 1000 and do over it.
        #objective (x-mu) + (x-mu)*(1-NORM_var)
        print("REACHED select best and worst")
        predictions_list = []
        available_indices = set(range(len(self.plan_dataset))).difference(self.indices_used)
        if quick_select:
            print("Doing QUICKSELECT FOR best and worst plan selection")
            available_indices = random.sample(list(available_indices), num_check_to_select)

        for single_formatted_plan_idx in available_indices:
            single_formatted_plan_struct = self.plan_dataset[single_formatted_plan_idx]
            encoded_plan = np.zeros(self.CONFIRMED_features_dimension)
            for single_feature in single_formatted_plan_struct[1]:
                temp_tuple_feature = tuple(single_feature)
                if temp_tuple_feature in self.CONFIRMED_features:
                    encoded_plan[self.RBUS_indexing.index(temp_tuple_feature)] = 1
            #end for loop
            try:
                predictions = self.learning_model_bayes.get_outputs_from_distribution(encoded_plan,
                                                                                      num_samples=NUM_SAMPLES_KDE)
                mean_prediction = np.mean(predictions)
                prediction_variance = np.var(predictions)
            except:
                mean_prediction = 0.0
                prediction_variance = 0.0

            #todo NOTE USING MEAN PREDICTION, makes more sense with using variance for decisions
            predictions_list.append( (single_formatted_plan_idx,(mean_prediction,prediction_variance)) )


        max_variance = max(predictions_list,key = lambda x:x[1][1])[1][1]
        mean_prediction =  sum([x[1][0] for x in predictions_list])/len(predictions_list)
        #end for loop through the formatted plan indices
        #SORT by (x-mu) + (x-mu)*(1-NORM_var)
        if max_variance == 0: #can happen in some cases
            predictions_list = sorted(predictions_list,
                                      key=lambda x: x[1][0] - mean_prediction )
        else:
            predictions_list = sorted(predictions_list,
                            key = lambda x: x[1][0]-mean_prediction + (x[1][0]-mean_prediction)*(1-x[1][1]/max_variance))
        #end else
        #now we take the top and bottom num_samples/2
        chosen_indices = [x[0] for x in predictions_list[0:int(num_samples/2)]] +\
                            [x[0] for x in predictions_list[-int(num_samples / 2):]]

        print("Test plans mean prediction, and prediction variance")
        print(predictions_list[0:int(num_samples/2)] + predictions_list[-int(num_samples/2):])

        return [self.plan_dataset[x] for x in chosen_indices]



# ================================================================================================
