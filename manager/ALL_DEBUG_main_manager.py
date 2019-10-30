# from itertools import combinations
# from pyclustering.cluster.kmedoids import kmedoids
# from pyclustering.utils import timedcall, metric
# from scipy import integrate
# from scipy.stats import describe as summ_stats_fnc
# #---------------------------------------------------------
# from plan_generator.journey_world import Journey_World
# from plan_generator.journey_world_planner import Journey_World_Planner
# from feedback.oracle import oracle
# from multiprocessing import Pool
# import numpy as np
# import pickle
# import random
# import copy
# import math
# from scipy.stats import norm
# from linetimer import CodeTimer
# # from common.system_logger.general_logger import logger
# from learning_engine.bayesian_linear_model import bayesian_linear_model
# from selection_engine import RBUS_selection
# import re
#
# """
# IMPORTANT NOTES:
#
# 1) We parallelized the RBUS integration step
# 2) WE hard drop those plans that have the same encoding as any of the previous plans. THIS MAKES SENSE.
# if they were high freq features that were different, they would not have the same encoding. ALSO if new features come to light
# then in a future round they will not have the same encoding
# 3) SEE the bayes linear model file. for learnings in that area
#
# """
# #===============================================================
#
# EXPECTED_NOISE_VARIANCE = 1.0  #the mean is 0
#
# INFINITESIMAL_VALUE = 1e-15
#
# #===============================================================
#
# # --------------
# def get_biModal_gaussian_gain_function(a=0.2, b=0.8, sd=0.1):
#     """
#     :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
#     There is no special property with probabilities and summing to 1 in this application. Only relative gain values
#     that match what we need.
#     :param x_value:
#     :param a:
#     :param b:
#     :return: the function
#     """
#     gain_a = norm(a, sd)
#     gain_b = norm(b, sd)
#     return lambda x: (gain_a.pdf(x) + gain_b.pdf(x)) / 2
#
#
# # ================================================================================================
#
# def get_gain_function(min_value, max_value, lower_percentile=0.1, upper_percentile=0.9):
#     """
#     :summary: abstract gain function
#     :param x_value:
#     :return:
#     :return:
#     """
#     range = max_value - min_value
#     lower_fifth = min_value + range * lower_percentile
#     upper_fifth = min_value + range * upper_percentile
#     scaled_std_dev = range * 0.1
#     return get_biModal_gaussian_gain_function(lower_fifth, upper_fifth, scaled_std_dev)
#
#
# # ================================================================================================
#
# def string_char_split(input_string):
#     """
#
#     :param input_string:
#     :return:
#     """
#     ret_obj = []
#     ret_obj.extend(input_string)
#     return ret_obj
#
# #===============================================================
#
# def look_for_ordered_feature(feature_char_list, plan_char_seq):
#     """
#
#     :param feature_char_list:
#     :param plan_char_seq:
#     :return:
#     """
#     found_state = True
#     for single_char in feature_char_list:
#         if single_char not in plan_char_seq:
#             found_state = False
#             break
#         #end if
#         plan_char_seq = plan_char_seq[plan_char_seq.index(single_char)+1:]
#     #end for
#     return found_state
#
#
# #===============================================================
# def insert_liked_disliked_features(all_plans,liked_features,disliked_features):
#     """
#
#     :param all_plans:
#     :param liked_features:
#     :param disliked_features:
#     :return:
#     """
#     #todo NOTE cannot just check the features in the formatted dict because rare features below cutoff will not be there
#     formatted_plans_struct = []
#     for single_plan in all_plans:
#         found_like_features = []
#         for single_liked in liked_features:
#             if look_for_ordered_feature(single_liked,single_plan):
#                 found_like_features.append(string_char_split(single_liked))
#         found_dislike_features = []
#         for single_disliked in disliked_features:
#             if look_for_ordered_feature(single_disliked,single_plan):
#                 found_dislike_features.append(string_char_split(single_disliked))
#         formatted_plans_struct.append( [single_plan,found_like_features,found_dislike_features] )
#     #end outermost for
#     return formatted_plans_struct
#
#
# #===============================================================
#
# def planTrace_clustering(start_medoids, data_points, tolerance=0.25, show=False):
#     """
#     :param start_medoids: is a list of indices, which correspond to the medoids
#     :param data_points:
#     :param tolerance:
#     :param show:
#     :return:
#     """
#
#     # todo note start mediods is represented by the data point index
#     chosen_metric = metric.distance_metric(metric.type_metric.MANHATTAN)
#     kmedoids_instance = kmedoids(data_points, start_medoids, tolerance, metric=chosen_metric)
#     (ticks, result) = timedcall(kmedoids_instance.process)
#     print("execution time in ticks = ",ticks)
#
#     clusters = kmedoids_instance.get_clusters()
#     medoids = kmedoids_instance.get_medoids()
#
#     print("----finished clustering")
#
#     # if show is True:
#     #     visualizer = cluster_visualizer_multidim()
#     #     visualizer.append_clusters(clusters, data_points, 0)
#     #     visualizer.append_cluster([data_points[index] for index in start_medoids], marker='*', markersize=15)
#     #     visualizer.append_cluster(medoids, data=data_points, marker='*', markersize=15)
#     #     visualizer.show()
#
#     return clusters,medoids
#
# #===============================================================
#
# class Manager:
#     """
#     The Manager is tasked with handling the whole pipeline. It starts with creating the domain, generating the set of plans,
#     applying DBS, then taking feedback from oracle, then applying R-BUS and DBS. Then conducting further rounds. It will
#     take care of creating the appropriate data structures for each stage. If we want to recreate a run, then we can the same
#     seed.
#     """
#     RATIO_PLANS_EXPLORATION = 0.2
#     def __init__(self,
#                  num_rounds = 2,
#                  num_dataset_plans = 10000,
#                  with_simulated_human = True,
#                  max_feature_size = 5,
#                  FEATURE_FREQ_CUTOFF = 0.05,
#                  test_set_size = 1000,
#                  plans_per_round = 30 ,
#                  use_features_seen_in_plan_dataset = True,
#                  prob_per_level = (0.75,0.25),
#                  pickle_file_for_plans_pool = "default_plans_pool.p",
#                  relevant_features_prior_weights = (0.1, -0.1),
#                  random_seed = 18):
#         """
#         creates the domain, plan_generator, and the oracle.
#         :param num_rounds: for AL
#         :param num_dataset_plans: number of backlog plans
#         :param with_simulated_human: Or real human(for later use)
#         :param relevant_features_prior_weights: prior weights set up for liked and disliked features, first element for the
#         liked feature
#         """
#         self.completed = False
#         self.plans_per_round = plans_per_round #plans_per_roud used for the web interface
#         self.num_backlog_plans = num_dataset_plans
#         self.curr_round = 0 # -1th round is for the test set , 0th round is exploratory
#         self.num_rounds = num_rounds*2-1 #every even round is RBUS, 0 is DBS
#         self.max_feature_size = max_feature_size
#         self.learning_model_bayes = bayesian_linear_model()
#         self.model_MLE = None # will be set later
#         self.relevant_features_prior_weights = relevant_features_prior_weights
#         self.min_rating = 1e10 #extreme starting values that will be updated after first round of feedback.
#         self.max_rating = -1e10
#         self.indices_used = set()
#         self.random_seed = random_seed
#         self.sim_human = None
#
#         with open(pickle_file_for_plans_pool,"rb") as src:
#             self.plan_dataset = pickle.load(src)
#
#         self.all_s1_features = set()
#         for plan in self.plan_dataset:
#             for step in plan:
#                 for feature_key in step.keys():
#                     self.all_s1_features.add(feature_key)
#         if with_simulated_human:
#             gaussian_noise_sd = 0.0
#             print("SIMULATED HUMAN has probabilities", prob_per_level, " AND gaussian noise sd = ",gaussian_noise_sd)
#             self.sim_human = oracle(self.all_s1_features, probability_per_level=prob_per_level, gaussian_noise_sd=gaussian_noise_sd, seed=random_seed, rating_distribution="power_law") #todo pass modules into the manager instead of instantiating it inside
#
#         # formatted plan dataset -- plans in the agreed upon format; plan-ids are indexes of the plans in this dataset
#         # DBS and R-BUS, happen in this dataset
#         self.formatted_plan_dataset = []
#         needs_conversion = type(list(self.plan_dataset)[0][1]) != dict #helps distinguish if plans are action seq, or state seq
#         for single_plan in self.plan_dataset:
#             state_seq = single_plan
#             if needs_conversion:
#                 state_seq = self.planner.convert_actionSeq_to_StateSeq(single_plan)
#             feature_sequence = []
#             for step in state_seq:
#                 non_feature_keys_set = set()#set(["x","y"])
#                 for feature_key in set(step.keys()).difference(non_feature_keys_set):
#                     feature_sequence.append(feature_key)
#                     self.all_s1_features.add(feature_key)
#                 #end inner for loop
#             #end for loop through state seq
#             all_features = []
#             for i in range(1,max_feature_size+1):
#                 all_features += list(combinations(feature_sequence, i))
#             #end for loop
#             feature_string = "".join(feature_sequence)
#             formatted_plan = [state_seq, all_features,feature_string]
#             # all_iterables = []
#             # for i in range(1,max_feature_size+1):
#             #     all_iterables.append(combinations(feature_sequence, i))
#             # #end for loop
#             # formatted_plan = [state_seq, all_iterables]
#             self.formatted_plan_dataset.append(formatted_plan)
#         self.annotated_plans = []
#         self.annotated_plans_by_round = []
#         #todo NOTE THE FOLLOWING DICT SHOULD NOT BE USED FOR TRAINING.
#         self.previously_annot_plans_dict = {} #maps feature string to full annotated plan.
#         self.switch_to_diversity_sampling = False #set to true when the plan scores for selection by freq hit 0
#         self.sorted_plans = []
#         self.score_dict = {}
#         self.seen_features = set()
#         self.subsumed_smaller_bigger_dict = {}#maps to the parent feature
#         self.subsumed_score_dict = {}#maps from the parent feature to the children
#         # format used - [[state-sequence], [list of features], [liked features], [disliked features], rating]
#         # this will be updated once the user annotates the presented plans
#         # after each round
#
#         #important data structures
#         # maps feature to index in the vector to be used for DBS/RBUS purpose. So take a plan, and encode it using this
#         # data structure when you want to do DBS/RBUS
#         self.RBUS_indexing = []
#         self.liked_features = set()
#         self.disliked_features = set()
#                     # print("---BIG ERROR-- RIGHT NOW ALL THE FEATURES ARE KNOWN AT THE START")
#                     # for single_feature in self.sim_human.feature_preferences_dict.keys():
#                     #     entry =  self.sim_human.feature_preferences_dict[single_feature]
#                     #     if entry[0] == "like":
#                     #         self.liked_features.add(single_feature)
#                     #     else:
#                     #         self.disliked_features.add(single_feature)
#         self.relevant_features = self.liked_features.union(self.disliked_features)
#         self.relevant_features_dimension = len(self.relevant_features)
#         self.RBUS_indexing = sorted(list(self.relevant_features))
#         self.RBUS_prior_weights = None
#         self.RBUS_prior_weights = [0.0]*self.relevant_features_dimension
#         self.num_cores_RBUS = 5
#
#         #now compute the features, scores, and plan sorting
#         self.average_feat_distance = 0.0
#         self.tried_indices = set()
#         self.feature_freq_scoring(FEATURE_FREQ_CUTOFF)
#         self.order_plans()
#         self.test_set = None# Do not extract test set here !! wait until you have a trained model and then pull interesting plans to test
#         #the test set should be well balanced between points inside the interesting region, and outside of it
#         self.plans_for_subseq_rounds = [] # will be initialized by the subprocess function that handles the manager
#         self.results_by_round = []
#     # ================================================================================================
#
#     def compute_freq_dict(self, formatted_plan_dataset, features_set):
#         """
#         :summary:
#         :param formatted_plan_dataset: at index 1, must have the sequence of features in the plan
#         :param features_set:
#         :return:
#         """
#         dataset_size = len(formatted_plan_dataset)
#         freq_dict = {x: 0 for x in features_set}
#         for plan in formatted_plan_dataset:
#             plan_features = plan[1]
#             for feature_seq_tuple in features_set:
#                 if self.check_for_feature(feature_seq_tuple,plan_features):
#                     freq_dict[feature_seq_tuple] += 1 / dataset_size  # this was the freq is automatically calculated at the end
#             # end for loop through features
#         # end for loop through plans
#         return freq_dict
#
#
#     # ================================================================================================
#     def feature_freq_scoring(self,FREQ_CUTOFF = 0.20):
#         """
#         :summary : find all features greater than the cutoff and score each feature by it's frequency
#         :return:
#         """
#
#         # the following code is done so that you can have tuples of size 1
#         base_s1_features = [(x,) for x in self.all_s1_features]
#         new_set_features_to_measure = base_s1_features
#         while len(new_set_features_to_measure) > 0:
#             with CodeTimer():
#                 freq_results_dict = self.compute_freq_dict(self.formatted_plan_dataset, new_set_features_to_measure)
#             freq_results_items = freq_results_dict.items()
#             filtered_freq_results = [x for x in freq_results_items if x[1] > FREQ_CUTOFF]
#             filtered_features_set = set([x[0] for x in filtered_freq_results])
#             filtered_score_dict = dict(filtered_freq_results)
#             self.score_dict = {**(self.score_dict),**filtered_score_dict}  # new dict merge in python 3.5. Shallow merge (one level), with 2nd dict getting precedence
#             new_set_features_to_measure = set()
#             for precursor_feature in filtered_features_set: #do NOT need to ALSO do all_s1->filtered because a+bc will be in ab+c
#                 for successor_feature in base_s1_features:
#                     new_feature = tuple(list(precursor_feature) + list(successor_feature))
#                     new_set_features_to_measure.add(new_feature)
#                 #end inner for
#             #end outer for
#         # end while loop
#         print("BEFORE subsuming checks")
#         print("The score dict keys ",sorted(list(self.score_dict.keys())))
#         print("The num keys",len(self.score_dict.keys()))
#         # remove subsumed features
#         # todo NOTE this was important because (by example) if ab and cd and klm were freq and in a plan, and we scored ab and cd
#         #  then that plan would be chosen over a plan with abcd which was also a (less) frequent pattern. but abcd subsumes ab and cd.
#         #  we would kill two birds with one stone. HOWEVER if another plan has a better pattern and also ab cd (separately), then the
#         #  score for abcd should go down. Hence we track the subsumption dict. CAN BE SUBSUMED BY MORE THAN ONE !
#         all_feature_seq = self.score_dict.keys()
#         for primary_feature_tuple_seq in all_feature_seq:
#             for compared_feature_tuple_seq in all_feature_seq:
#                 if primary_feature_tuple_seq == compared_feature_tuple_seq:  # and note that there should NOT be any duplicates.
#                     continue
#                 if len(compared_feature_tuple_seq) < len(primary_feature_tuple_seq):
#                     continue #cannot be a subsequence
#                 primary_feature_len = len(primary_feature_tuple_seq)
#                 compared_feature_len = len(compared_feature_tuple_seq)
#                 #check for substring matching at any of the possible start indices
#                 if any((all(primary_feature_tuple_seq[j] == compared_feature_tuple_seq[i + j] for j in range(primary_feature_len))
#                                                 for i in range(compared_feature_len - primary_feature_len + 1))):
#                     try:
#                         self.subsumed_smaller_bigger_dict[primary_feature_tuple_seq].append(compared_feature_tuple_seq)
#                     except KeyError:
#                         self.subsumed_smaller_bigger_dict[primary_feature_tuple_seq] = [compared_feature_tuple_seq]
#
#                 # end if all the elements in the features match
#             # end for loop through the compared features
#         # end for loop through the primary features
#
#         features_to_subsume = list(self.subsumed_smaller_bigger_dict.keys())
#         for feature in features_to_subsume:
#             subsuming_features = self.subsumed_smaller_bigger_dict[feature]
#             for single_subsuming_feature in subsuming_features:
#                 if single_subsuming_feature not in features_to_subsume: #this check ensures that we do not increase the score of ab in the case <a,ab,abc>
#                     self.score_dict[single_subsuming_feature] += self.score_dict[feature]  # INCREASE THE SCORE
#             self.subsumed_score_dict[feature] = self.score_dict[feature]
#             del self.score_dict[feature]
#         print("AFTER subsuming checks")
#         print("The score dict keys ",sorted(list(self.score_dict.keys())))
#         print("The num keys",len(self.score_dict.keys()))
#
#     # ---end function
#
#     # ================================================================================================
#     def order_plans(self):
#         """
#         :param self:
#         :return:
#         """
#         sorted_plans = []  # should contain the PLAN, FEATURES, SCORE
#         for plan_idx in range(len(self.formatted_plan_dataset)):
#             plan = self.formatted_plan_dataset[plan_idx]
#             curr_score = 0.0
#             plan_features = plan[1]
#             for feature_seq_tuple in self.score_dict.keys():
#                 if self.check_for_feature(feature_seq_tuple,plan_features):
#                     curr_score += self.score_dict[feature_seq_tuple]
#             # end for through features
#             sorted_plans.append([plan, plan_idx, curr_score])  # the plan and features are the same in this problem
#         # end for loop through plans
#         self.sorted_plans = sorted(sorted_plans, key=lambda x: x[-1], reverse=True)
#
#     # ================================================================================================
#     def conduct_rounds(self):
#         """
#
#         :return:
#         """
#
#         #TODO NOT YET USED OR TESTED
#         pass
#
#     # ================================================================================================
#     def store_annot_test_set(self,annot_test_set):
#         """
#         :return:
#         """
#
#         filtered_plans = []
#         for plan in annot_test_set:
#             if plan[0] != None:
#                 plan[2] = ["".join(x) for x in plan[2]]
#                 plan[3] = ["".join(x) for x in plan[3]]
#                 plan[4] = float(plan[4])
#                 filtered_plans.append(plan)
#         # end for loop
#         #we do NOT update the indices, if there are new features these new features are unused for predictions
#         self.test_set = filtered_plans
#
#     # ================================================================================================
#     def get_next_round_plans_for_intf(self):
#         """
#
#         :return:
#         """
#
#         next_round_plans = self.plans_for_subseq_rounds[0]
#         self.plans_for_subseq_rounds = self.plans_for_subseq_rounds[1:]
#         #additionally for each plan, insert the previously annotated features (separate for liked and disliked)
#         # todo NOTE cannot just check the features in the formatted dict because rare features below cutoff will not be there
#         next_round_plans = insert_liked_disliked_features(next_round_plans,self.liked_features,self.disliked_features)
#         print(next_round_plans)
#         return next_round_plans
#
#     # ================================================================================================
#     def prep_next_round_plans_for_intf(self):
#         """
#         :return:
#         """
#
#         #todo NOTE THIS FUNCTION is called by the subprocess function that handles the manager
#         print("PREPPING plans for round ", self.curr_round)
#         chosen_plans = []
#         if self.curr_round > self.num_rounds:
#             print("SENDING TEST SET PLANS printout from main manager = ",self.test_set)
#             chosen_plans = self.test_set
#         if self.curr_round == 0:
#             chosen_plans = self.sample_by_DBS(self.plans_per_round)
#         elif self.curr_round % 2 != 0:
#             print("prepped DBS half of plans")
#             chosen_plans = self.sample_by_DBS(int(self.plans_per_round * Manager.RATIO_PLANS_EXPLORATION))
#         else: #it is an RBUS round = even rounds after 0 (0 case is handled by the first if)
#             print("prepped RBUS half of plans")
#             chosen_plans = self.sample_by_RBUS(int(self.plans_per_round * (1- Manager.RATIO_PLANS_EXPLORATION)))
#         #---now package the plans in a string format that can be consumed by the web interface
#         self.curr_round += 1
#         self.plans_for_subseq_rounds.append([string_char_split(x[2]) for x in chosen_plans]) #the 2nd index is for the feature string. split into list of chars
#
#     # ================================================================================================
#     def check_next_round_RBUS(self):
#         """
#         :return:
#         """
#         print(self.curr_round, self.curr_round % 2 == 0 and self.curr_round != 0)
#         return self.curr_round % 2 == 0 and self.curr_round != 0
#
#
#     # ================================================================================================
#     def encode_by_relevant_features(self,formatted_plan_idx):
#         """
#
#         :param formatted_plan_idx:
#         :return:
#         """
#
#         plan_features = self.formatted_plan_dataset[formatted_plan_idx][1]
#         plan_encoding = np.zeros(self.relevant_features_dimension, dtype=float)
#         for single_feature in plan_features:
#             if single_feature in self.relevant_features:
#                 plan_encoding[self.RBUS_indexing.index(single_feature)] = 1
#         return plan_encoding
#
#     # ================================================================================================
#     @staticmethod
#     def check_for_feature(feature_seq_tuple, all_features):
#         """
#         :param num_test_plans:
#         :return:
#         """
#         return feature_seq_tuple in all_features
#
#
#
#     # ================================================================================================
#
#     def extract_test_set(self,num_test_plans):
#         """
#
#         :param num_test_plans:
#         :return:
#         """
#         random.seed(self.random_seed)
#         test_set_indices = random.sample(range(len(self.formatted_plan_dataset)),num_test_plans)
#         #remove indices from available indices
#         self.indices_used.update(test_set_indices)
#         return [self.formatted_plan_dataset[x] for x in test_set_indices]
#
#
#     # ================================================================================================
#
#     def sample_by_DBS(self, num_samples=10):
#         """
#         :summary: Does the frequency analysis and finds all features greater than cutoff frequency.
#         NO SIZE LIMIT ! or can update code to specify if time constraints are pressing
#         :param num_samples:
#         :return:
#         """
#
#         if self.switch_to_diversity_sampling:
#             print("-------SWITCHING TO DIVERISTY SAMPLING IN DBS")
#             return self.sample_randomly_wDiversity(num_samples)
#         num_plans_needed = num_samples
#         chosen_plans = []
#         while len(chosen_plans) < num_plans_needed:  # iterate through the list
#             while True:
#                 if self.sorted_plans[0][1] in self.indices_used:
#                     self.sorted_plans = self.sorted_plans[1:]
#                     continue #next iteration of the loop, DO NOT CONTINUE in the same iteration. The next one could be used too
#                 #end if
#                 plan = self.sorted_plans[0][0]  # FIRST rescore (may have changed features in first plan)
#                 curr_score = 0.0
#                 plan_feature_list = plan[1]
#                 features_in_curr_plan = []
#                 for single_feature in self.score_dict.keys():
#                     if single_feature in self.seen_features:
#                         continue
#                     if single_feature in plan_feature_list :
#                         features_in_curr_plan.append(single_feature)
#                         curr_score += self.score_dict[single_feature]
#                 # end for loop through features
#                 self.sorted_plans[0][-1] = curr_score
#                 if self.sorted_plans[0][-1] >= self.sorted_plans[1][-1]:
#                     self.seen_features.update(features_in_curr_plan)
#                     break#no need to sort further (for now)
#                 # else insert it back into the list
#                 # Todo apparently good code requires one to create a user defined object with a comparison function
#                 # before using python's bisenct.insort() in order to insert into a list. No extra lambda option is there.
#                 # the argument is that the calling of extra lambda functions multiple times is poor code and speed.
#                 # for now, I handcoded it
#                 upper_idx = 0
#                 lower_idx = len(self.sorted_plans) - 1
#                 # curr score is already in memory
#                 prev_mid_idx = 0
#                 while True:
#                     mid_idx = int((upper_idx + lower_idx) / 2)
#                     if curr_score >= self.sorted_plans[mid_idx][-1]:
#                         lower_idx = mid_idx
#                     else:
#                         upper_idx = mid_idx
#                     if mid_idx == prev_mid_idx:
#                         if curr_score > self.sorted_plans[lower_idx][-1]:
#                             self.sorted_plans.insert(lower_idx, self.sorted_plans[0])
#                         else:
#                             self.sorted_plans.insert(lower_idx + 1, self.sorted_plans[0])
#                         del self.sorted_plans[0]  # remove the top copy
#                         break
#                     else:
#                         prev_mid_idx = mid_idx
#                 # end while true for inserting into sorted list
#             # end while true for selecting the next top plan
#
#             # THUS FAR we have only removed the features from the score dict,
#             #WE ALSO need to reduce the score of subsumed (smaller) features FROM the bigger features that subsume them
#             single_chosen_plan = self.sorted_plans[0][0]
#             chosen_plan_features_list = single_chosen_plan[1]
#             seen_and_subsumed = []
#             for single_subsumed_feature_tupleSeq in self.subsumed_smaller_bigger_dict.keys():
#                 if single_subsumed_feature_tupleSeq in chosen_plan_features_list:
#                     seen_and_subsumed.append(single_subsumed_feature_tupleSeq)
#                     bigger_features = self.subsumed_smaller_bigger_dict[single_subsumed_feature_tupleSeq]
#                     for single_bigger_feature in bigger_features:
#                         try:
#                             #this will only be done once as the subsumed feature will be removed from the dict after that.
#                             #so the score of the bigger feature will go only as low as it's true frequency.
#                             self.score_dict[single_bigger_feature] -= self.subsumed_score_dict[single_subsumed_feature_tupleSeq]
#                         except KeyError: #this can happen when the bigger feature itself is subsumed. eg <a,ab,abc>
#                             pass
#             #end for loop through subsumed features
#             for single_feature in seen_and_subsumed: #remove these from the subsumed dict, so we dont repeat looking at it.
#                 del self.subsumed_smaller_bigger_dict[single_feature]
#                 del self.subsumed_score_dict[single_feature]
#             #end for loop
#             chosen_plans.append(copy.deepcopy(single_chosen_plan))
#             self.indices_used.add(self.sorted_plans[0][1])
#             # print("plan selected score = ", self.sorted_plans[0][-1])
#             self.sorted_plans = self.sorted_plans[1:]
#             if self.sorted_plans[0][-1] == 0.0:
#                 print("SWITCHING to random sampling with diversity ")
#                 self.switch_to_diversity_sampling = True
#                 break;
#         # end while loop when not enough plans are in the chosen plan list
#         if self.switch_to_diversity_sampling:
#             return chosen_plans + self.sample_randomly_wDiversity(num_samples-len(chosen_plans))
#         else:
#             return chosen_plans
#
#     # ================================================================================================
#     def compute_plan_feature_distance(self,plan_a_features,plan_b_features):
#         """
#             #includes dropping subsumed features, and CHECKING for seen features
#         :return:
#         """
#         # If the different features are subsumed, then ignore.
#         different_features = set(plan_a_features).difference(set(plan_b_features))
#         drop_features = set()
#         for single_feat in different_features:
#             if single_feat in self.seen_features:
#                 drop_features.add(single_feat)
#                 continue
#             try:
#                 subsuming_feats_list = self.subsumed_smaller_bigger_dict[single_feat]
#                 if any(x in plan_b_features for x in subsuming_feats_list):
#                     drop_features.add(single_feat)
#             except KeyError:
#                 pass
#         #end for loop
#         return len(different_features.difference(drop_features))
#
#
#     # ================================================================================================
#     def compute_average_feature_distance(self):
#         """
#         :summary: compute the average distance (in number of features) between plans.
#         :return:
#         """
#         total_distance = 0.0
#         for plan_a in self.formatted_plan_dataset:
#             for plan_b in self.formatted_plan_dataset:
#                 if plan_a == plan_b:
#                     continue
#                 total_distance += self.compute_plan_feature_distance(plan_a[1],plan_b[1])
#         #end inner for loop
#         n_val = len(self.formatted_plan_dataset)-1
#         self.average_feat_distance = total_distance/(n_val*(n_val+1)/2)
#
#
#     # ================================================================================================
#     def sample_randomly_wDiversity(self, num_samples):
#         """
#         Take plan if greater than avg feature distance over all other seen plans.
#         :param num_samples:
#         :return:
#         """
#         chosen_plans = []
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         available_indices.difference(self.tried_indices)
#         while len(chosen_plans) < num_samples:
#             curr_index = random.sample(available_indices,1)[0]
#             curr_plan = self.formatted_plan_dataset[curr_index]
#             avg_feat_distance = self.average_feat_distance #this is to handle the case when no seen indices are there
#             for single_seen_index in self.indices_used:
#                 avg_feat_distance += \
#                     self.compute_plan_feature_distance(curr_plan[1],self.formatted_plan_dataset[single_seen_index][1])
#             #end for loop
#             avg_feat_distance /= len(self.indices_used)+1 #+1 is for the starting point of "avg_feat_distance"
#             if avg_feat_distance >= self.average_feat_distance:
#                 chosen_plans.append(curr_plan)
#                 self.indices_used.add(curr_index)
#             else:
#                 self.tried_indices.add(curr_index)
#             #dont repeat this index
#             available_indices.remove(curr_index)
#         #end while loop
#         return chosen_plans
#
#
#
#
#     # ================================================================================================
#     def sample_randomly(self, num_samples):
#         """
#         Sample Plans randomly to test for oracle feedback and R-BUS strategy
#         sampled plans are considered used
#         :param num_samples:
#         :return:
#         """
#
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         new_indices_sampled = random.choices(list(available_indices), k=num_samples)
#         self.indices_used.update(new_indices_sampled)
#         return [self.formatted_plan_dataset[x] for x in new_indices_sampled]
#
#     # ================================================================================================
#
#
#
#
#     @staticmethod
#     def APPROX_parallel_variance_computation(input_list):
#         """
#
#         :param input_list:
#         :return:
#         """
#
#
#         # -----end inner function get_biModal_gaussian_gain_function
#
#         learning_model_bayes, encoded_plan, min_rating, max_rating, include_gain = input_list
#         preference_possible_values = np.linspace(min_rating, max_rating, num=100)
#         predictions, kernel = learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan,
#                                                                                          num_samples=500)
#         gain_function = get_gain_function(min_rating, max_rating)
#         preference_prob_density = kernel(preference_possible_values)
#         normalizing_denom = np.sum(preference_prob_density)
#         mean_preference = np.sum(preference_prob_density * preference_possible_values) / normalizing_denom
#         preference_variance = np.sum(
#             np.square(preference_possible_values - mean_preference) * preference_prob_density) / normalizing_denom
#         composite_func_integral = 0.0
#         if include_gain:  # saves time when we do not use gain function, otherwise we only need the variance
#             gain_outputs = np.array([gain_function(x) for x in preference_possible_values])
#             composite_func_outputs = preference_prob_density * gain_outputs
#             # composite_func_outputs = composite_func_outputs.reshape(composite_func_outputs.shape[0])
#             composite_func_integral = np.sum(composite_func_outputs* preference_possible_values)
#         # print("TEMP PRINT: The composite_func_integral of the prob*gain over the outputs is =", composite_func_integral)
#         # this composite_func_integral is the Expected gain from taking this sample
#         return predictions, composite_func_integral, preference_variance
#
#     # ================================================================================================
#     @staticmethod
#     def parallel_variance_computation(input_list):
#         """
#
#         :param input_list:
#         :return:
#         """
#         def get_gain_function(min_value, max_value, lower_percentile = 0.2, upper_percentile = 0.8):
#             """
#             :summary: abstract gain function
#             :param x_value:
#             :return:
#             :return:
#             """
#             range = max_value - min_value
#             lower_fifth = min_value + range * lower_percentile
#             upper_fifth = min_value + range * upper_percentile
#             scaled_std_dev = range * 0.1
#             return get_biModal_gaussian_gain_function(lower_fifth, upper_fifth, scaled_std_dev)
#         # --------------
#         def get_biModal_gaussian_gain_function(a=0.2, b=0.8, sd=0.1):
#             """
#             :summary: It averages two gaussians at means "a" and "b" who have the same std deviation.
#             There is no special property with probabilities and summing to 1 in this application. Only relative gain values
#             that match what we need.
#             :param x_value:
#             :param a: IS NOT PERCENTILE, but absolute values
#             :param b:
#             :return: the function
#             """
#             gain_a = norm(a, sd)
#             gain_b = norm(b, sd)
#             return lambda x: (gain_a.pdf(x) + gain_b.pdf(x)) / 2
#         #-----end inner function get_biModal_gaussian_gain_function
#
#         learning_model_bayes,encoded_plan, min_rating, max_rating, include_gain  = input_list
#         min_rating = min_rating #sample from a large spread
#         max_rating = max_rating
#         preference_possible_values = np.linspace(min_rating, max_rating, num=100)
#         predictions, kernel = learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan,
#                                                                                               num_samples=500)
#         gain_function = get_gain_function(min_rating,max_rating)
#         preference_prob_density = kernel(preference_possible_values)
#         normalizing_denom = np.sum(preference_prob_density)
#         if normalizing_denom == 0.0:
#             normalizing_denom = 1.0 # THIS IS VERY HACKY and needed to avoid nan, that occurs when there are no features, and alpha is deterministic
#         #todo NOTE this is HACKY way of computing the mean value. It is not at all mathematically valid, but very fast and works well
#         mean_preference = np.sum(preference_prob_density * preference_possible_values) / normalizing_denom
#         preference_variance = np.sum(np.square(preference_possible_values - mean_preference) * preference_prob_density) / normalizing_denom
#         if np.isnan(preference_variance):
#             print("CATCH CASE")
#         composite_func_integral = 0.0
#         if include_gain:  # saves time when we do not use gain function
#             gain_outputs = np.array([gain_function(x) for x in preference_possible_values])
#             composite_func_outputs = preference_prob_density * gain_outputs
#             composite_func_outputs = composite_func_outputs.reshape(composite_func_outputs.shape[0])
#             composite_func_integral = integrate.trapz(composite_func_outputs, preference_possible_values)
#         # print("TEMP PRINT: The composite_func_integral of the prob*gain over the outputs is =", composite_func_integral)
#         # this composite_func_integral is the Expected gain from taking this sample
#         return predictions,composite_func_integral, preference_variance
#
#     # ================================================================================================
#     def sample_for_SAME_features(self, num_samples=10):
#         """
#         :summary :
#         Sample such that we get the plan with the most number of the same features
#         :return:
#         """
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         sorted_plans = []  # should contain the PLAN, FEATURES, SCORE
#         for plan_idx in available_indices:
#             plan = self.formatted_plan_dataset[plan_idx]
#             curr_score = 0.0
#             plan_features = plan[1]
#             for single_plan_feature in plan_features:
#                 if single_plan_feature in self.seen_features:
#                     curr_score += 1
#             # end for through features
#             sorted_plans.append([plan, plan_idx, curr_score])  # the plan and features are the same in this problem
#         # end for loop through plans
#         sorted_plans = sorted(sorted_plans, key=lambda x: x[-1], reverse=True)
#         return [self.formatted_plan_dataset[x] for x in  [x[1] for x in sorted_plans[0:num_samples]] ]
#
#     # ================================================================================================
#     def set_num_cores_RBUS(self,num_cores):
#         """
#         """
#         self.num_cores_RBUS = num_cores
#
#     # ================================================================================================
#     def sample_by_RBUS(self, num_samples = 30, include_gain = True, include_feature_distinguishing = True):
#         """
#         :summary :
#         For each of the remaining annotated data:  Use the model learned thus far, and get output distribution.
#         Multiply by the output by the information value (gain) function.  for each data point
#         Then return the highest "n" samples based on this computed value
#         :return:
#         """
#         print("RBUS RUNNING")
#         if len(self.annotated_plans) == 0:
#             return self.sample_randomly_wDiversity(num_samples)
#
#         num_subProcess_to_use = self.num_cores_RBUS
#         # gain_function = RBUS_selection.get_gain_function(min_value=self.min_rating,max_value=self.max_rating)
#         #TODO remove all the print statements and integral checks, will speed things up considerably
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         # print("num available points = ",len(available_indices))
#         index_value_list = [] # a list of tuples of type (index,value)
#         with CodeTimer():
#             p = Pool(num_subProcess_to_use)
#             all_parallel_params = []
#             for single_plan_idx in available_indices:
#                 current_plan = self.formatted_plan_dataset[single_plan_idx][1]
#                 encoded_plan = np.zeros(self.relevant_features_dimension)
#                 for single_feature in current_plan:
#                     if single_feature in self.relevant_features:
#                         encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
#
#                 all_parallel_params.append([self.learning_model_bayes,encoded_plan,self.min_rating,self.max_rating,include_gain])
#             #end for loop through the available indices
#             # Note the last parameter below is a LIST, each entry is for a new process
#             results = p.map(self.parallel_variance_computation,all_parallel_params)#Note the last parameter is a LIST, each entry is for a new process
#             # results = [self.parallel_variance_computation(x) for x in all_parallel_params]
#             print("size of results =",len(results))
#             for single_idx_result in zip(available_indices,results):
#                 single_idx = single_idx_result[0]
#                 single_result = single_idx_result[1]
#                 predictions, composite_func_integral, preference_variance = single_result
#                 # preference_prob_density = kernel(preference_possible_values)
#                 # preference_variance = np.var(preference_prob_density)
#                 # composite_func_integral = 0.0
#                 # if include_gain: #saves time when we do not use gain function
#                 #     gain_outputs = np.array([gain_function(x) for x in preference_possible_values])
#                 #     composite_func_outputs = preference_prob_density*gain_outputs
#                 #     composite_func_outputs = composite_func_outputs.reshape(composite_func_outputs.shape[0])
#                 #     composite_func_integral = integrate.trapz(composite_func_outputs, preference_possible_values)
#                 # # print("TEMP PRINT: The composite_func_integral of the prob*gain over the outputs is =", composite_func_integral)
#                 # #this composite_func_integral is the Expected gain from taking this sample
#                 index_value_list.append((single_idx,composite_func_integral,preference_variance))
#         #end codetimer profiling section
#         #---NOW we have to select top n plans such that every successive plan selected also considers diversity w.r.t to the previous plans selected
#         #the best plan is determined by normalized gain * normalized variance
#
#         if include_gain:
#             gain_array = np.array([x[1] for x in index_value_list])
#         else:
#             gain_array = np.array([1.0 for x in index_value_list])
#         norm_gain_array = gain_array/np.max(gain_array) #normalize it
#         variance_array = np.array([x[2] for x in index_value_list])
#         norm_variance_array = variance_array/ np.max(variance_array)  # normalize it
#         norm_gain_variance_array = [norm_gain_array[x]*norm_variance_array[x] for x in range(len(norm_gain_array))]
#         #now store (idx,norm_gain*norm_variance)
#         if include_feature_distinguishing:
#             # todo NOTE this version below does score+score/numFeatures. do NOT need abs because it is always a +ve value. score is an integral
#             index_value_list = [(index_value_list[x][0],
#                                 norm_gain_variance_array[x]+norm_gain_variance_array[x]/len(self.formatted_plan_dataset[x][1]))
#                                 for x in range(len(index_value_list))]
#         else: #just the score and NOT sc + sc/|Feat|
#             index_value_list = [(index_value_list[x][0],norm_gain_variance_array[x]) for x in range(len(index_value_list))]
#         #NOTE the order of ENTRIES in index value list will now be fixed
#         indices_list = tuple([x[0] for x in index_value_list]) #to map plan_index (entry) to the physical index(position)
#         #see the use of indices list a little further down in code.
#         encoded_previous_plans = []
#         chosen_indices = []
#         index_value_list = sorted(index_value_list, key=lambda x: x[1],reverse=True)
#         curr_index_in_sorted_list = 0
#         while len(chosen_indices) < num_samples and len(index_value_list) > curr_index_in_sorted_list:
#             curr_max = index_value_list[curr_index_in_sorted_list]
#             curr_index_in_sorted_list += 1
#             current_plan_index = curr_max[0]
#             current_plan_encoding = self.encode_by_relevant_features(current_plan_index)
#             #check if this has the same relevant feature encoding as prev plans
#             duplicate_by_relevant_encoding = False
#             for single_prev_plan_enc  in encoded_previous_plans:
#                 if np.array_equal(current_plan_encoding,single_prev_plan_enc):
#                     # todo NOTE: ?? <UNSURE> hard dropping all plans with same relevant features seems to have us choosing less important plans with high likelihood ??
#                     duplicate_by_relevant_encoding = True
#                     break
#                 #end if
#             #end for
#             if not duplicate_by_relevant_encoding:
#                 chosen_indices.append(current_plan_index)
#                 encoded_previous_plans.append(current_plan_encoding)
#             #end if
#         #end while
#         chosen_normGainVar_values = [norm_gain_variance_array[indices_list.index(x)] for x in chosen_indices]
#         print("TEMP PRINT chosen norm_E[gain]*norm_var values (with diversity) = ",chosen_normGainVar_values)
#         print("Overall statistics for CHOSEN norm_E[gain]*norm_var are ", summ_stats_fnc(chosen_normGainVar_values))
#         print("Overall statistics for ALL norm_E[gain]*norm_var are ", summ_stats_fnc(norm_gain_variance_array))
#         self.indices_used.update(chosen_indices)
#         addendum = []
#         # we may NOT have enough samples from rbus if there are no more plans without duplicate encoding
#         if len(chosen_indices) < num_samples:
#             print("RBUS had too few plans w/o duplicate known feature encoding, so adding random samples w diversity of size ",num_samples-len(chosen_indices))
#             addendum = self.sample_randomly_wDiversity(num_samples-len(chosen_indices))
#
#         print("SAMPLES CHOSEN IN RBUS ",chosen_indices)
#         return addendum + [self.formatted_plan_dataset[x] for x in chosen_indices]
#
#         # todo KEEP THIS CODE
#         # LATER do soft dropping where we reweight for diversity and choose based on E(gain) AND diversity
#         # ---------------------
#         # index_value_list = tuple(index_value_list)
#         # num_remaining_plans = len(index_value_list)
#         # indices_list = tuple([x[0] for x in index_value_list])
#         # norm_gain_array = norm_gain_array/np.max(norm_gain_array) #normalize it
#         # distances_array = np.zeros(num_remaining_plans)
#         # plans_array = np.zeros((num_remaining_plans,self.relevant_features_dimension))#the each plan is along a row, the columns are the relevant features
#         # for idx in range(num_remaining_plans): #FILL the plans array
#         #     current_plan_idx = indices_list[idx]#this IS CORRECT, we need the normal idx as insertion position in plans array
#         #     plans_array[idx] = self.encode_by_relevant_features(current_plan_idx)
#         # #end for
#         # chosen_indices = []
#         # chosen_indices.append(max(index_value_list,key=lambda x:x[1])[0])
#         # while len(chosen_indices) < num_samples:
#         #     last_chosen_plan = self.formatted_plan_dataset[chosen_indices[-1]]
#         #     #compute distance with all other plans
#         #     #and add distance between this plan and all other plans to the index_distanceSum_list
#         #     tiled_plan_array = np.tile(last_chosen_plan,(num_remaining_plans,1))
#         #     difference_array = np.abs(plans_array-tiled_plan_array)
#         #     add_distances_array = np.sum(difference_array,axis=1) #sum across the columns (axis =1) (within each row)
#         #     distances_array += add_distances_array
#         #     norm_dist_array = distances_array/np.max(distances_array)
#         #     score_array = norm_dist_array*norm_gain_array
#         #     chosen_indices.append(np.where(score_array==np.max(score_array))[0])
#         # #end while loop
#
#     # ================================================================================================
#     def reformat_features_and_update_indices(self, annotated_plans):
#         """
#
#         :param annotated_plans:
#         :return:
#         """
#         filtered_plans = []
#         for plan in annotated_plans:
#             if plan[0] != None:
#                 plan[2] = ["".join(x) for x in plan[2]]
#                 plan[3] = ["".join(x) for x in plan[3]]
#                 try:
#                     plan[4] = float(plan[4]) #handles empty string and incorrect format string.
#                 except ValueError:
#                     plan[4] = float(0.0)
#                 filtered_plans.append(plan)
#         #end for loop
#         self.update_indices(filtered_plans)
#
#     # ================================================================================================
#     def update_indices(self, annotated_plans):
#         """
#         Takes annotated plans and update indices required for R-BUS and DBS modules
#         :param annotated_plans: format - [[state-sequence],
#         [list of features], [liked features], [disliked features], rating]
#         :return:
#         """
#         self.annotated_plans_by_round.append(annotated_plans)
#         liked_features = set()
#         disliked_features = set()
#         for single_plan in annotated_plans:
#             liked_features.update(single_plan[2])
#             disliked_features.update(single_plan[3])
#             if len(liked_features) + len(disliked_features) == 0:
#                 continue #disregard this plan
#             self.annotated_plans.append(single_plan)
#         #end for loop through annotated plans
#         self.liked_features.update(liked_features)
#         self.disliked_features.update(disliked_features)
#
#         #Now reindex the features after the removals
#         self.relevant_features = self.liked_features.union(self.disliked_features)
#         self.relevant_features_dimension = len(self.relevant_features)
#         self.RBUS_indexing = sorted(list(self.relevant_features))
#         # print("Temporary print RBUS index list",self.RBUS_indexing)
#         # todo we could define the prior weights based on the PREVIOUS MODEL trained in the previous round
#         # define prior weights based on what's liked and disliked
#         self.RBUS_prior_weights = np.zeros(self.relevant_features_dimension)
#         for single_feature in self.relevant_features:
#             if single_feature in self.liked_features:
#                 self.RBUS_prior_weights[self.RBUS_indexing.index(single_feature)] = \
#                     self.relevant_features_prior_weights[0]
#             else:
#                 self.RBUS_prior_weights[self.RBUS_indexing.index(single_feature)] = \
#                     self.relevant_features_prior_weights[1]
#
#     #================================================================================================
#     def get_feedback(self,plans):
#         """
#
#         :param plans:
#         :return:
#         """
#         left_over_plans = []
#         annot_plans = []
#         for single_plan in plans:
#             try:
#                 annot_plans.append(self.previously_annot_plans_dict[single_plan[2]])
#             except KeyError:
#                 left_over_plans.append(single_plan)
#
#         #todo SAVE all the feedback annotations and ratings, so the noise is consistent across all the methods
#         if len(left_over_plans) > 0:
#             print("ANNOTATION NEW PLANS of size ",len(left_over_plans))
#         newly_annot_plans = self.sim_human.get_feedback(left_over_plans)
#         annot_plans += newly_annot_plans
#         for single_new_annot_idx in range(len(newly_annot_plans)):
#             self.previously_annot_plans_dict[left_over_plans[single_new_annot_idx][2]] = newly_annot_plans[single_new_annot_idx]
#         sorted_annot_plans = sorted(annot_plans, key = lambda x:x[-1])
#         print("FEEDBACK GIVEN WAS =", sorted_annot_plans)
#
#         return annot_plans
#     #================================================================================================
#     def store_all_plans_feedback(self):
#         """
#         :return:
#         """
#         remaining_plans = [self.formatted_plan_dataset[x] for x in range(len(self.formatted_plan_dataset)) if x not in self.indices_used]
#         self.get_feedback(remaining_plans)
#
#
#
#
#     # ================================================================================================
#     def relearn_model(self, learn_LSfit = False, num_chains=1):
#         """
#         since we have the relevant features and some annotated plans(<plans, rating>, we learn a liner regression model
#         by Bayesian Learning. The manager will connect to the learning engine to learn and update the model
#         :return:
#         """
#         if len(self.annotated_plans) == 0: #this can happen when the user has not rated anything and just clicked
#             return #nothing to learn from
#
#
#         rescaled_plans = copy.deepcopy(self.annotated_plans)
#         #todo NOTe rescaling is currently removed but we still need to track min and max rating
#
#         ratings = np.array([x[-1] for x in rescaled_plans])
#         min_rating = np.min(ratings)
#         max_rating = np.max(ratings)
#         if min_rating < self.min_rating:
#             self.min_rating = min_rating
#         if max_rating > self.max_rating:
#             self.max_rating = max_rating
#         scaler = self.max_rating - self.min_rating
#         min_rating = self.min_rating
#         max_rating = self.max_rating
#
#         scores = []
#         for plan in self.annotated_plans:
#             scores.append(plan[-1])
#
#         #todo NOTE no rescaling needed and is REMOVED . The user just enters whatever range of values they like to express preference.
#         # We only try to be accurate about the bottom and top 20%
#         # for single_plan in rescaled_plans:
#         #     single_plan[-1] = single_plan[-1] / scaler - self.min_rating / scaler#last term is to make the min == 0
#         # # end for loop
#
#         encoded_plans_list = []
#         for single_plan in rescaled_plans:
#             encoded_plan = [np.zeros(self.relevant_features_dimension), single_plan[4]]
#
#             for single_feature in single_plan[2] + single_plan[3]: #the liked and disliked features
#                 encoded_plan[0][self.RBUS_indexing.index(single_feature)] = 1
#             encoded_plans_list.append(encoded_plan)
#
#         MLE_reg_model = None
#         if learn_LSfit:
#             from sklearn import linear_model
#             # MLE_reg_model = linear_model.LinearRegression(fit_intercept=True) #NORMALIZE wont help, the input is binary. Already normalized
#             # MLE_reg_model = linear_model.LinearRegression(fit_intercept=False) #NORMALIZE wont help, the input is binary. Already normalized
#             MLE_reg_model = linear_model.Ridge(fit_intercept=False) #normalize wont help here either, the input is binary, already normalized
#             input_dataset = np.array([x[0] for x in encoded_plans_list])
#             output_dataset = np.array([x[1] for x in encoded_plans_list])
#
#             #todo PLEASE test the weighted fit more on toy problems. When you normalized the weights, it failed miserably
#             # which was unexpected
#             # WEIGHTS ARE CURRENTLY TURNED OFF, VERY ERRATIC PERFORMANCE
#             # weight_func = RBUS_selection.get_gain_function(self.min_rating,self.max_rating)
#             # weights = [weight_func(x) for x in output_dataset]
#             # weights = [1.0 for x in output_dataset]
#
#             MLE_reg_model.fit(input_dataset, output_dataset)
#             print("Coefficients's values ", MLE_reg_model.coef_)
#             print("Intercept: %.4f" % MLE_reg_model.intercept_)
#             self.model_MLE = MLE_reg_model
#         #end if learn_LSfit
#
#         self.learning_model_bayes.learn_bayesian_linear_model(encoded_plans_list,
#                                                               self.RBUS_prior_weights,
#                                                               self.relevant_features_dimension,
#                                                               sd= EXPECTED_NOISE_VARIANCE,
#                                                               sampling_count=2000,
#                                                               num_chains=num_chains)
#
#
#         if self.sim_human != None:  #i.e. we are in simulated testing
#
#             param_stats = [self.learning_model_bayes.linear_params_values["betas"][0:2000, x] for x in
#                           range(self.relevant_features_dimension)]
#             param_stats = [summ_stats_fnc(x) for x in param_stats]
#             bayes_feature_dict = copy.deepcopy(self.sim_human.feature_preferences_dict)
#             for single_feature in bayes_feature_dict:
#                 try:
#                     bayes_feature_dict[single_feature].append(param_stats[self.RBUS_indexing.index(single_feature)])
#                 except ValueError: # for the feature not being in the list
#                     pass
#             MLE_feature_dict = copy.deepcopy(self.sim_human.feature_preferences_dict)
#             for single_feature in MLE_feature_dict:
#                 try:
#                     MLE_feature_dict[single_feature].append(MLE_reg_model.coef_[self.RBUS_indexing.index(single_feature)])
#                 except ValueError:  # for the feature not being in the list
#                     pass
#             #todo not just mean, do MODE of features
#             print("Bayes params ","  ||  ".join([str(x) for x in bayes_feature_dict.items()]))
#             print("Bayes intercept summ stats")
#             print(summ_stats_fnc(self.learning_model_bayes.linear_params_values["alpha"][0:2000]))
#             print("MLE params ","  ||  ".join([str(x) for x in MLE_feature_dict.items()]))
#             # self.evaluate(self.test_set)
#
#     #================================================================================================
#     def evaluate(self,annotated_test_plans=None):
#         """
#         :summary : we have the actual result in the annotated test plans, we see what the model
#         would have predicted , which is a distribution over values. We take the mode of the output distribution
#         as the prediction. Then the error is RMS error over the test set
#
#         :param annotated_test_plans:
#         :return:
#         """
#         if annotated_test_plans == None:
#             annotated_test_plans = self.test_set
#
#         #TODO NOTE WE FILTER THE PLANS WITH NO FEATURES ARE RATED 0.0 (not interesting) ONLY FOR TESTING, NOT FOR TRAINING (UNKNOWN THEN)
#         annotated_test_plans = [x for x in annotated_test_plans if x[-1] != 0.0]
#
#         MLE_total_squared_error = 0.0
#         MLE_error_list = []
#         bayes_total_squared_error = 0.0
#         bayes_error_list = []
#         bayes_target_prediction_list = []
#         MLE_final_error = 0.0
#         MLE_target_prediction_list = []
#         bayes_final_error = 0.0
#
#         count_samples = 0
#
#
#         #end if
#         print(" IN EVALUATION ",self.curr_round ,self.num_rounds+1)
#         if self.curr_round > self.num_rounds+1: #then all rounds were completed
#             self.completed = True
#
#         print("=====EVALUATION===== after round ",self.curr_round/2-1)#recall we double the round numbers to handle dbs and rbus separation, except for first round which is only dbs.
#         for single_annot_plan_struct in annotated_test_plans:
#
#             current_plan_features = single_annot_plan_struct[2] + single_annot_plan_struct[3]
#             encoded_plan = np.zeros(self.relevant_features_dimension)
#             count_samples +=1
#             for single_feature in current_plan_features:
#                 temp_tuple_feature = tuple(single_feature)
#                 if temp_tuple_feature in self.relevant_features:
#                     encoded_plan[self.RBUS_indexing.index(temp_tuple_feature)] = 1
#
#             true_value = float(single_annot_plan_struct[-1])
#             #end for loop
#             if self.model_MLE != None:
#                 mle_predict = self.model_MLE.predict([encoded_plan])[0]
#                 current_squared_error = math.pow(true_value - mle_predict, 2)
#                 MLE_total_squared_error += current_squared_error
#                 MLE_error_list.append(math.sqrt(current_squared_error))
#                 MLE_target_prediction_list.append((true_value,mle_predict))
#
#             predictions,kernel = self.learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan, num_samples=500)
#             preference_possible_values = np.linspace(self.min_rating, self.max_rating, num=100)
#             preference_prob_density = kernel(preference_possible_values)
#             if not np.min(preference_prob_density) == np.max(preference_prob_density):
#                 index_mode= np.where(preference_prob_density == np.max(preference_prob_density))[0][0]
#                 mode_prediction = preference_possible_values[index_mode]
#                 normalizing_denom = np.sum(preference_prob_density)
#                 mean_prediction = np.sum(preference_prob_density * preference_possible_values)/normalizing_denom
#                 prediction_variance = np.sum(np.square(preference_possible_values-mean_prediction)*preference_prob_density)/normalizing_denom
#             else:
#                 #absolute uniform distribution over all preference values, actually means we have no information,
#                 # we SET the alpha to be fixed, so no variations in the output.
#                 mode_prediction = 0.0
#                 mean_prediction = 0.0
#                 prediction_variance = 0.0
#             #todo NOTE USING MEAN PREDICTION, makes more sense with using variance for decisions
#             current_squared_error = math.pow(true_value - mean_prediction, 2)
#             bayes_total_squared_error += current_squared_error
#             bayes_error_list.append( math.sqrt(current_squared_error))
#             bayes_target_prediction_list.append((true_value, mode_prediction, mean_prediction, prediction_variance))
#
#
#         if count_samples == 0:
#             print("NOT TEST PLANS WERE RATED FAILED")
#             return bayes_final_error,MLE_error_list
#
#         #end for loop
#         if self.model_MLE != None:
#             MLE_final_error = math.sqrt(MLE_total_squared_error / count_samples)
#             print("LINEAR MODEL The average error in ALL regions is = ", MLE_final_error)
#             print("LINEAR MODEL Error Statistics of ALL regions, ", summ_stats_fnc(MLE_error_list))
#             print("LINEAR MODEL target and prediction ", MLE_target_prediction_list)
#
#         #end if
#         bayes_final_error = math.sqrt(bayes_total_squared_error / count_samples)
#         print("BAYES MODEL The average error in ALL regions= ", bayes_final_error)
#         print("BAYES MODEL Error Statistics of ALL regions , ",summ_stats_fnc(bayes_error_list))
#         print("BAYES MODEL target and prediction ",bayes_target_prediction_list)
#         if self.results_by_round == None:
#             self.results_by_round = []
#         self.results_by_round = self.results_by_round.append([bayes_final_error,MLE_final_error])
#         print("bayes error list ", bayes_error_list)
#         print("MLE error list ", MLE_error_list)
#
#         return bayes_final_error,MLE_final_error
#
#     # ================================================================================================
#
#     def get_balanced_test_set_before_training(self, test_set_size=1000, ratio_interesting_region=0.4):
#         """
#         :summary : extract plans such that we have a portion of the plans in the interesting region
#         :param test_set_size:
#         :param ratio_interesting_region:
#         :return:
#         plans are selected like
#            ||xxxxx.....******|*******.....xxxxx|| where x is interesting region, * is just above and below the middle, and "." is in between
#         """
#
#         # in the following line we div by 2 since we want plans with ratings at either extreme
#         per_region_num_interesting_plans = int(test_set_size * ratio_interesting_region / 2)
#         num_trivial_plans = test_set_size - 2 * per_region_num_interesting_plans
#         # the above trivial plans are split into two groups as well just below the preferred mark, and just above the rejected mark as in the function comments
#         # TODO remove all the print statements and integral checks, will speed things up considerably
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         # print("num available points = ",len(available_indices))
#         index_value_list = []  # a list of tuples of type (index,value)
#         with CodeTimer():
#             # compute and order the predicted scores for the remaining plans
#             self.sim_human.change_rating_noise(0.0) #the balanced dataset depends on true ratings
#             rated_annot_plans = self.sim_human.get_feedback(self.formatted_plan_dataset)
#             rated_annot_plans = enumerate(rated_annot_plans,0) #start from zero
#             sorted_all_results = sorted(rated_annot_plans, key=lambda x: x[1][-1], reverse=True)
#             num_results = len(sorted_all_results)
#             # get plans from top, bottom, and middle
#             print("******EXPECTED RATINGS in interesting regions*********")
#             print(sorted_all_results[0:per_region_num_interesting_plans] + sorted_all_results[
#                                                                            -per_region_num_interesting_plans:])
#             # todo ADD diversity to the samples chosen. Avoid the same RELEVANT feature encoding. May not be enough then though ??
#             chosen_indices = [x[0] for x in
#                               sorted_all_results[0:per_region_num_interesting_plans]]  # most preferred value
#             chosen_indices += [x[0] for x in
#                                sorted_all_results[int(num_results / 2 - num_trivial_plans / 2):int(
#                                    num_results / 2)]]  # just above median
#             chosen_indices += [x[0] for x in
#                                sorted_all_results[-per_region_num_interesting_plans:]]  # most hated/least preferred
#             chosen_indices += [x[0] for x in
#                                sorted_all_results[int(num_results / 2):int(
#                                    num_results / 2 + num_trivial_plans / 2)]]  # just below median
#             # sort the results are get plans on either end
#         self.indices_used.update(chosen_indices)
#         return [self.formatted_plan_dataset[x] for x in chosen_indices]
#
#     # ================================================================================================
#
#     def get_balanced_test_set_after_training(self, test_set_size = 1000, ratio_interesting_region = 0.4):
#         """
#         :summary : extract plans such that we have a portion of the plans in the interesting region
#         :param test_set_size:
#         :param ratio_interesting_region:
#         :return:
#         plans are selected like
#            ||xxxxx.....******|*******.....xxxxx|| where x is interesting region, * is just above and below the middle, and "." is in between
#         """
#
#
#         # in the following line we div by 2 since we want plans with ratings at either extreme
#         per_region_num_interesting_plans = int(test_set_size*ratio_interesting_region/2)
#         num_trivial_plans = test_set_size - 2*per_region_num_interesting_plans
#         #the above trivial plans are split into two groups as well just below the preferred mark, and just above the rejected mark as in the function comments
#         #TODO remove all the print statements and integral checks, will speed things up considerably
#         available_indices = set(range(len(self.formatted_plan_dataset)))
#         available_indices.difference_update(self.indices_used)
#         # print("num available points = ",len(available_indices))
#         index_value_list = [] # a list of tuples of type (index,value)
#         with CodeTimer():
#             #compute and order the predicted scores for the remaining plans
#             all_results = []
#             for single_plan_idx in available_indices:
#                 current_plan = self.formatted_plan_dataset[single_plan_idx][1]
#                 encoded_plan = np.zeros(self.relevant_features_dimension)
#                 for single_feature in current_plan:
#                     if single_feature in self.relevant_features:
#                         encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
#                 #end for loop through current plan
#                 #the last false is for including gain, we do not care about that for output prediction
#                 predictions, kernel = self.learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan,
#                                                                                                  num_samples=500)
#                 preference_possible_values = np.linspace(self.min_rating, self.max_rating, num=100)
#                 preference_prob_density = kernel(preference_possible_values)
#                 index_mode = np.where(preference_prob_density == np.max(preference_prob_density))[0][0]
#                 mode_prediction = preference_possible_values[index_mode]
#                 all_results.append((single_plan_idx,mode_prediction))
#             #end for loop through the available indices
#             sorted_all_results = sorted(all_results,key= lambda x:x[1],reverse=True)
#             num_results = len(sorted_all_results)
#             #get plans from top, bottom, and middle
#             print("******EXPECTED RATINGS in interesting regions*********")
#             print(sorted_all_results[0:per_region_num_interesting_plans] + sorted_all_results[-per_region_num_interesting_plans:])
#             #todo ADD diversity to the samples chosen. Avoid the same RELEVANT feature encoding. May not be enough then though ??
#             chosen_indices = [x[0] for x in sorted_all_results[0:per_region_num_interesting_plans]] #most preferred value
#             chosen_indices += [x[0] for x in
#                 sorted_all_results[ int(num_results/2-num_trivial_plans/2):int(num_results/2)]] #just above median
#             chosen_indices += [x[0] for x in sorted_all_results[-per_region_num_interesting_plans:]] #most hated/least preferred
#             chosen_indices += [x[0] for x in
#                 sorted_all_results[int(num_results/2):int(num_results/2+num_trivial_plans/2)]] #just below median
#             #sort the results are get plans on either end
#         return [self.formatted_plan_dataset[x] for x in chosen_indices]
#
#
# #=============================================================================
#     def region_based_evaluation(self, annotated_test_plans, eval_percentile_regions=[(0.0, 0.1), (0.9, 1.0)],
#                                 inside_region=True):
#         """
#
#         :param annotated_test_plans:
#         :param eval_percentile_regions:
#         :return:
#         """
#
#         #TODO NOTE WE FILTER THE PLANS WITH NO FEATURES ARE RATED 0.0 (not interesting) ONLY FOR TESTING, NOT FOR TRAINING (UNKNOWN THEN)
#         annotated_test_plans = [x for x in annotated_test_plans if x[-1] != 0.0]
#         bayes_total_squared_error = 0.0
#         bayes_error_list = []
#         MLE_total_squared_error = 0.0
#         bayes_target_prediction_list = []
#         MLE_target_prediction_list = []
#         MLE_error_list = []
#         # convert the percentiles to actual regions
#         sorted_ratings = sorted([x[-1] for x in annotated_test_plans])
#         num_plans = len(annotated_test_plans)
#         cutoff_regions = []
#         for single_region in eval_percentile_regions:
#             bound_indices = [int(x * num_plans) for x in single_region]
#             if num_plans in bound_indices:
#                 bound_indices[bound_indices.index(num_plans)] = num_plans - 1
#             cutoff_regions.append([sorted_ratings[x] for x in bound_indices])
#
#         # ratings_range = self.max_rating - self.min_rating
#         # cutoff_regions = [(self.min_rating + x[0]*ratings_range, self.min_rating + x[1]*ratings_range) for x in eval_percentile_regions]
#         count_samples = 0
#
#         print(
#             "NOTE WE ASSUME A PLAN WITH NO KNOWN FEATURES IS OF VALUE 0, AND SO NOT COUNTED IN THE TEST SET EVALUATION")
#         print("ALL RATINGS are = ", sorted_ratings)
#         print("Cutoffregions = ", cutoff_regions)
#         for single_annot_plan_struct in annotated_test_plans:
#             true_rating = single_annot_plan_struct[-1]
#             region_checks = [x[0] <= true_rating and true_rating <= x[1] for x in cutoff_regions]
#             if inside_region and not True in region_checks:
#                 continue
#             if not inside_region and True in region_checks:
#                 continue
#             count_samples += 1
#             current_plan = single_annot_plan_struct[1]
#             encoded_plan = np.zeros(self.relevant_features_dimension)
#             for single_feature in current_plan:
#                 if single_feature in self.relevant_features:
#                     encoded_plan[self.RBUS_indexing.index(single_feature)] = 1
#             # ---first do the simple MLE model, easier
#
#             true_value = float(single_annot_plan_struct[-1])
#             if self.model_MLE != None:
#                 mle_predict = self.model_MLE.predict([encoded_plan])[0]
#                 current_squared_error = math.pow(
#                     single_annot_plan_struct[-1] - mle_predict, 2)
#                 MLE_total_squared_error += current_squared_error
#                 MLE_error_list.append(math.sqrt(current_squared_error))
#                 MLE_target_prediction_list.append((true_value,mle_predict))
#
#             # ---now do the bayes model, need to get the MODE prediction
#             predictions, kernel = self.learning_model_bayes.get_outputs_and_kernelDensityEstimate(encoded_plan,
#                                                                                                   num_samples=500)
#             preference_possible_values = np.linspace(self.min_rating, self.max_rating, num=100)
#             preference_prob_density = kernel(preference_possible_values)
#             if not np.min(preference_prob_density) == np.max(preference_prob_density):
#                 index_mode = np.where(preference_prob_density == np.max(preference_prob_density))[0][0]
#                 mode_prediction = preference_possible_values[index_mode]
#                 normalizing_denom = np.sum(preference_prob_density)
#                 mean_prediction = np.sum(preference_prob_density * preference_possible_values) / normalizing_denom
#                 prediction_variance = np.sum(np.square(
#                     preference_possible_values - mean_prediction) * preference_prob_density) / normalizing_denom
#             else:
#                 # absolute uniform distribution over all preference values, actually means we have no information,
#                 # we SET the alpha to be fixed, so no variations in the output.
#                 mode_prediction = 0.0
#                 mean_prediction = 0.0
#                 prediction_variance = 0.0
#             #todo NOTE USING MEAN PREDICTION, makes more sense with using variance for decisions
#             current_squared_error = math.pow(true_value - mean_prediction, 2)
#             bayes_total_squared_error += current_squared_error
#             bayes_error_list.append( math.sqrt(current_squared_error))
#             bayes_target_prediction_list.append((true_value, mode_prediction, mean_prediction, prediction_variance))
#
#         # end for loop
#         print(" If inside INTERESTING REGION is ", inside_region)
#         if count_samples == 0:
#             print("THERE WERE ZERO SAMPLES ACCORDING TO OUR CURRENT RATING MAX!!")
#             print("This can happen when the training set has a higher upper bound than the test set")
#             # todo INSTEAD OF DOING TOP 20% BY MAX RANGE, should do upper 20% OF POINTS. BUT THIS DEFEATS THE PURPOSE, SO NO
#             # WE AIM TO FIND TOP 20% AND BOTTOM 20% BY RANGE, not by points.
#             return 0.0
#
#         print("NUM SAMPLES = ", count_samples, " for when INTERESTING REGION IS", inside_region)
#         if self.model_MLE != None:
#             MLE_final_error = math.sqrt(MLE_total_squared_error / count_samples)
#             print("LINEAR MODEL The average REGION error is = ", MLE_final_error, "for percentile regions ",
#                   eval_percentile_regions)
#             print("LINEAR MODEL Error Statistics of CHOSEN regions , ", summ_stats_fnc(MLE_error_list))
#             print("LINEAR MODEL target and prediction ", MLE_target_prediction_list)
#
#         # end if
#         bayes_final_error = math.sqrt(bayes_total_squared_error / count_samples)
#         print("BAYES MODEL The average REGION error is = ", bayes_final_error, "for percentile regions ",
#               eval_percentile_regions)
#         print("BAYES MODEL Error Statistics of CHOSEN regions , ", summ_stats_fnc(bayes_error_list))
#         print("BAYES MODEL target and prediction ",bayes_target_prediction_list)
#
#         return bayes_final_error,MLE_final_error
#
# # ================================================================================================
