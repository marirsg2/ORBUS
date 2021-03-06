from itertools import combinations, permutations
import random
import re
from common.system_logger.general_logger import logger
import numpy as np

class oracle:
    #todo add the functionality where we can input the preference model for the human, just like before(XAIP version)
    """
    a random simulated human. It requires a Journey world for initialization. It will have some preferences for the
    features present in the grid. The default implementation only has l-1 and l-2 features, but it can have more if needed.
    It is supposed to take a plan, annotate and rate it.
    """

    def __init__(self,
                 all_s1_features,
                 l_n_max = 2,
                 probability_per_level = (0.75, 0.25),
                 like_probability = 0.5,
                 seed = 18,
                 rating_distribution ="uniform",
                 gaussian_noise_sd = 0.1,
                 verbose = False):
        """
        Using self.probability_per_level assigns what this human likes and dislikes, and what features are irrelevant
        :param l_n_max: the maximum level of features. Default at 2.
        :param probability_per_level: The probability of features that are relevant for each level of features
        :param like_probability: the probability with which the user will like a relevant feature
        :param seed: to have the same preference model
        """
        self.gaussian_noise_sd = gaussian_noise_sd
        self.max_level = l_n_max
        if len(probability_per_level) != l_n_max:
            raise Exception("Did not provide probability values for all the levels of features")

        if seed is not None:
            random.seed(seed)

        if gaussian_noise_sd != 0.0:
            logger.debug("INCLUDED gaussian noise in simulated human ratings%f " % gaussian_noise_sd)
        else:
            logger.debug("NO gaussian noise in simulated human ratings ")

        if rating_distribution == "uniform":
            rating_distribution = oracle.rating_default_dist
        elif rating_distribution == "power_law":
            rating_distribution = oracle.rating_power_law
            power_law_samples = 1 - np.random.power(5, 1000)
            self.power_law_sampled_points = power_law_samples
            # self.power_law_sampled_points = np.interp(power_law_samples,
            #                                           (power_law_samples.min(),
            #                                            power_law_samples.max()),
            #                                           (0, 0.4)) + 0.1


        self.s1_features = all_s1_features
        # now assigning features he likes and dislikes.
        self.feature_preferences_dict = {}

        feature_level_list = [l+1 for l in range(l_n_max)]
        for current_feature_level, relevance_probability in zip(feature_level_list, probability_per_level):
            features_at_current_level = permutations(self.s1_features, current_feature_level)
            for single_feature in features_at_current_level:
                r1 = random.random()
                if r1 <= relevance_probability:
                    # so this feature is relevant
                    r2 = random.random()
                    if r2 <= like_probability:
                        # so the user likes the feature
                        self.feature_preferences_dict[single_feature] = ["like",
                                                                        rating_distribution(self, "like")]
                    else:
                        self.feature_preferences_dict[single_feature] = ["dislike",
                                                                        rating_distribution(self, "dislike")]

    # ===============================================================================
    def rating_default_dist(self,sentiment):
        """
        :param sentiment:
        :return:
        """
        if sentiment == "like":
            return random.uniform(0.1, 0.5)
        return -random.uniform(0.1, 0.5)

    # ===============================================================================
    def rating_power_law(self, sentiment):
        if sentiment == "like":
            return random.choice(self.power_law_sampled_points)
        return -random.choice(self.power_law_sampled_points)

    #===============================================================================
    def get_feedback(self, plans):
        """
        The input plans are in standard format - [<state-seq>, <list of features>]
        :param plans: accepts a list of plans, and returns their annotated version. Will be used by R-BUS.
        :return: plans in annotated format [<state-seq>, <list of features>, <liked features>, <disliked features>, rating]
        """

        annotated_plans_list = []
        for single_plan in plans:
            all_plan_features = single_plan[1]
            plan_rating = 0.0 + random.gauss(mu = 0.0, sigma=self.gaussian_noise_sd)# we start with this, and add
            # each feature rating onto it, if that feature
            # is relevant to the user.
            liked_features = []
            disliked_features = []
            for single_feature in self.feature_preferences_dict:
                if single_feature in all_plan_features != None: #i.e. we found a match
                    # if it is present then its relevant to the preference model
                    feature_preference_info = self.feature_preferences_dict[single_feature]
                    if feature_preference_info[0] == "like":
                        liked_features.append(single_feature)
                    else:
                        disliked_features.append(single_feature)
                    plan_rating += feature_preference_info[1]
            annotated_plan = [single_plan[0],
                              single_plan[1],
                              liked_features,
                              disliked_features,
                              plan_rating]
            annotated_plans_list.append(annotated_plan)
            logger.debug("Plan Rated, uncapped rating: %.4f" % plan_rating)
        return annotated_plans_list
    # ===============================================================================
    # def OLD_get_feedback(self, plans):
    #     """
    #     The input plans are in standard format - [<state-seq>, <list of features>]
    #     :param plans: accepts a list of plans, and returns their annotated version. Will be used by R-BUS.
    #     :return: plans in annotated format [<state-seq>, <list of features>, <liked features>, <disliked features>, rating]
    #     """
    #
    #     annotated_plans_list = []
    #     for single_plan in plans:
    #         contained_features = single_plan[1]
    #         plan_rating = 0.5 + random.gauss(mu = 0.0, sigma=self.gaussian_noise_sd)# we start with this, and add
    #         # each feature rating onto it, if that feature
    #         # is relevant to the user.
    #         liked_features = []
    #         disliked_features = []
    #         for single_feature in contained_features:
    #             if single_feature in self.feature_preferences_dict:
    #                 # if it is present then its relevant to the preference model
    #                 feature_preference_info = self.feature_preferences_dict[single_feature]
    #                 if feature_preference_info[0] == "like":
    #                     liked_features.append(single_feature)
    #                 else:
    #                     disliked_features.append(single_feature)
    #                 plan_rating += feature_preference_info[1]
    #         annotated_plan = [single_plan[0],
    #                           single_plan[1],
    #                           liked_features,
    #                           disliked_features,
    #                           plan_rating]
    #         annotated_plans_list.append(annotated_plan)
    #         logger.debug("Plan Rated, uncapped rating: %.4f" % plan_rating)
    #     return annotated_plans_list
    # ===============================================================================
    def change_rating_noise(self,new_noise_sd):
        self.gaussian_noise_sd = new_noise_sd
        print("Simulated human RATING NOISE is CHANGED to ", new_noise_sd)
    # ===============================================================================
