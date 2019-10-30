import numpy as np
import copy
import random
import itertools

#=================================
#---global defines----
ANNOTATION_KEYWORD = "annotation"
POSITIVE_ANNOTATION_STRING = "up"
NEGATIVE_ANNOTATION_STRING = "down"
#=================================


def simple_cost(start_state, curr_plan):
    """

    :param start_state:
    :param curr_plan:
    :return:
    """
    total_cost = 0
    for step in curr_plan:
        total_cost += np.sum(np.abs(step))
    return total_cost
#=================================

def geometric_series_sum(init_value, ratio, series_length):
    """
    :param init_value:
    :param ratio:
    :param series_length: The length of series. The number "n". This can be an array too.
    :return:
    """
    return init_value*(1- np.power(ratio, series_length)) / (1 - ratio)

#=================================

def simple_feature_distance(plan_a, plan_b,ignore_features = [],feature_count_scaler = 0.5):
    """
    :summary : computes the abs(feature count difference) between plan a and b.
    Was written with binary features in mind, but may work for some cardinal features
    Each feature is counted separately. Then the geometric sum is taken with r = 0.5, a = 1.0
    So each features resultant count is scaled to be between [0,1]
    then we sum up and return the value
    :param plan_a:
    :param plan_b:
    :return:
    """
    dict_feature_to_index = {}
    feature_counts = []
    next_index = 0
    plan_a = copy.deepcopy(plan_a)
    plan_b = copy.deepcopy(plan_b)
    for step in plan_a:
        for single_feature in ignore_features:
            try:
                del step[single_feature]
            except KeyError:
                pass
        # end for through ignore features
        for feature in step.keys():
            try:
                feature_counts[dict_feature_to_index[feature]] += 1
            except KeyError:
                dict_feature_to_index[feature] = next_index
                next_index += 1
                feature_counts.append(1) #this features count is incremented
    #end for loop through plan a
    for step in plan_b:
        for single_feature in ignore_features:
            try:
                del step[single_feature]
            except KeyError:
                pass
        #end for through ignore features
        for feature in step.keys():
            try:
                feature_counts[dict_feature_to_index[feature]] -= 1 #we decrement, so the overall result is the difference
            except KeyError:
                dict_feature_to_index[feature] = next_index
                next_index += 1
                feature_counts.append(-1) #for plan b we count in the negative direction, to get the difference at the end
    #end for loop through plan b
    feature_counts = [abs(x) for x in feature_counts] # we only want the absolute value of the count difference. Symmetric distance
    scaled_feature_counts = geometric_series_sum(0.5,feature_count_scaler,feature_counts)
    return sum(scaled_feature_counts)

#======================================================

def simple_stepwise_average_plan_distance(plan_a, plan_b,max_distance = 1.0):
    """
    :summary : simply sum up stepwise distances. For the extra steps present in one plan and not in the other,
    we take the distance to THE LAST STATE of the shorter plan.
    :param plan_a:
    :param plan_b:
    :param max_distance: default is 1.0 to not make a difference. If set to the true max distance, then you will get
    a normalized average distance
    :return:
    """
    total_distance = 0
    diff_len = len(plan_b) - len(plan_a)
    if diff_len > 0:
        try:
            plan_a = plan_a + [plan_a[-1]]*diff_len
        except:
            print(plan_a)
    else: #if exception then plan is longer
        plan_b = plan_b + [plan_b[-1]]*(-diff_len) #diff len is negative in this case, so *-1
    for step_idx in range(len(plan_a)):#both plans are now equal length
        total_distance += np.linalg.norm([plan_a[step_idx]["x"]-plan_b[step_idx]["x"],
                                          plan_a[step_idx]["y"]-plan_b[step_idx]["y"]])/max_distance
    return total_distance/len(plan_a)

class Journey_World_Planner:

    def __init__(self, grid_cells_list):
        """
        Initializes the grid_array - i^th row j^th column element is vector with a feature turned to 1, if that feature
        is present in the corresponding position
        :param grid_cells_list:
        """
        # instance variables
        self.ignore_features = ["x", "y", "start", "goal"]
        self.index_to_feature_dict = None
        self.start_state_array = None
        self.max_x = None
        self.max_y = None
        self.max_cost = None
        self.max_distance = None
        self.num_features = None
        self.start_pos = None
        self.curr_pos = None
        self.goal_pos = None
        #----------------------
        self.transform_function = self.default_transform_function
        # initializing functions
        self.initialize_grid_array(grid_cells_list)

    # ==========================================================

    def default_transform_function(self, in_state, curr_pos, in_action):
        """

        :param action:
        :return:
        """
        return curr_pos + in_action  # elementwise sum of arrays

    # ==========================================================

    def default_avail_actions(self, grid_state, position):
        """

        :param state:
        :param position:
        :return:
        """
        avail_actions = []
        if position[0] == self.max_x-1: #-1 because it is zero indexed
            avail_actions += [(-1, 0)]
        elif position[0] == 1:
            avail_actions += [(1, 0)]
        else:
            avail_actions += [(-1, 0), (1, 0)]
        # now for y axis
        if position[1] == self.max_y-1: #-1 because it is zero indexed
            avail_actions += [(0, -1)]
        elif position[1] == 1:
            avail_actions += [(0, 1)]
        else:
            avail_actions += [(0, -1), (0, 1)]

        return avail_actions


    # ==========================================================
    def default_heuristic(self, curr_pos):
        """
        :summary : #sum up manhattan distance. expects each step in curr plan to be of form (<x step>,<y step>)
        like (0,-1)
        :param state:
        :return:
        """
        return np.sum(np.abs(curr_pos - self.goal_pos))

    # ==========================================================
    def initialize_grid_array(self, grid_cells_list):
        """
        :summary : Parse the state of the gridworld from the ui
        :return:
        """
        dimensions_set = set()
        max_x = 0
        max_y = 0
        state_struct = grid_cells_list
        for single_dict in state_struct:
            dimensions_set = dimensions_set.union(set(single_dict.keys()))
            if single_dict["x"] > max_x: max_x = single_dict["x"]
            if single_dict["y"] > max_y: max_y = single_dict["y"]
        # end for
        for feature in self.ignore_features:
            try:
                dimensions_set.remove(feature)
            except KeyError:
                pass
        total_num_dimens = len(dimensions_set)
        feature_indices_listMap = sorted(list(
            dimensions_set))  # the index of each feature is the index in the grid array vector at each x,y as well
        index_dict = {key: i for i, key in enumerate(feature_indices_listMap)}
        self.index_to_feature_dict = {i: key for i, key in enumerate(feature_indices_listMap)}
        # The grid is max_x and Max_y in size, but the indices start at 0 to max-1
        grid_array = np.zeros((max_x, max_y, total_num_dimens), dtype=int)
        for single_dict in state_struct:
            # find the x y position into which the dict features should be inserted
            curr_vector = grid_array[single_dict["x"] - 1, single_dict["y"] - 1, :]  # -1 because it is zero offset
            for single_key in single_dict.keys():
                # push the values from the dict into the grid array at the correct position (other than x,y) specd by
                # the index_dict
                if not (single_key in self.ignore_features):
                    curr_vector[index_dict[single_key]] = single_dict[single_key]

                if single_key == "start":
                    self.start_pos = np.array((single_dict["x"] - 1, single_dict["y"] - 1), dtype=int)
                    self.curr_pos = copy.deepcopy(self.start_pos)

                if single_key == "goal":
                    self.goal_pos = np.array((single_dict["x"] - 1, single_dict["y"] - 1), dtype=int)
                # end if
            # end for
        # end for
        self.start_state_array = grid_array
        # self.feature_to_index_listMap = feature_indices_listMap
        self.max_x = max_x
        self.max_y = max_y
        self.max_cost = 1.5 * (
                    max_x + max_y)  # assumes we use the simple cost of manhattan distance covered by steps in plan.
        self.max_distance = np.linalg.norm([max_x, max_y])
        self.num_features = total_num_dimens
    #==========================================================
    def states_match(self, state_a,state_b ):
        """
        :summary : Match exact values and wildcards
        :param state_a: numpy array for state a
        :param state_b: numpy array for state b
        :return:
        """
        #find all wildcard positions and set them all equal in both states and to be 0.
        #wild cards == -1.

        # get a mask for each that notes the wildcard positions and combine the masks
        state_a_mask = np.isin(state_a,[-1])
        state_b_mask = np.isin(state_b,[-1])
        wildcard_positions = state_a_mask*state_b_mask #element wise multiplication
        value_mask = np.logical_not(wildcard_positions)
        relevant_state_a = state_a * value_mask
        relevant_state_b = state_b * value_mask
        if np.equal(relevant_state_a ,relevant_state_b):
            return True
        return False
    #---end function states_match

    #===================================================

    def convert_plan_to_list_position_seq(self, single_plan):
        """
        :summary: plans are sequences of actions, convert them into a sequence of positions. Useful to message the UI
        :param single_plan:
        :return:
        """
        curr_state_seq = [list(self.start_pos)]
        for single_step in single_plan:
            curr_state_seq.append(list(self.transform_function(
                    self.start_state_array,curr_state_seq[-1], np.array(single_step) ) ))
        # end inner for
        return curr_state_seq

    #===================================================

    def convert_plan_to_array_position_seq(self, single_plan):
        """
        :summary: plans are sequences of actions, convert them into a sequence of positions.
        :param single_plan:
        :return:
        """
        curr_state_seq = [self.start_pos]
        for single_step in single_plan:
            curr_state_seq.append(self.transform_function(
                    self.start_state_array,curr_state_seq[-1], np.array(single_step) ) )
        # end inner for
        curr_state_seq = [tuple(x) for x in curr_state_seq]
        return curr_state_seq



    # ===================================================

    def get_quickly_n_plans(self, num_plans, rand_dev_prob= 0.1, num_duplicates_wait = 10, random_increase = 0.05):
        """
        :summary : Uses a greedy heuristic best-depth first search heuristic.
        If the count is not increasing after 10 tries, it increases the amount of randomness in the search
        :param num_plans:
        :return:
        """
        all_plans = set()
        curr_count = len(all_plans)
        duplicates_count = 0
        while curr_count < num_plans:
            all_plans.add(tuple(self.get_greedy_heuristic_depth_search_plan(rand_dev_prob=rand_dev_prob)))
            if () in all_plans:
                all_plans.remove(()) # as an empty plan might have been added #todo Check with RAM about this
            new_count = len(all_plans)
            if new_count == curr_count:
                duplicates_count += 1
                if duplicates_count == num_duplicates_wait:
                    rand_dev_prob += random_increase
            else:
                curr_count = new_count
        return all_plans

    # ===================================================


    def get_greedy_heuristic_depth_search_plan(self, heuristic_func=default_heuristic,cost_func = simple_cost,
                                               available_action_function=default_avail_actions, rand_dev_prob=0.1):
        """
        :summary : at each step greedily takes the best action as per the heuristic. No backtracking implemented
        as of yet
        :param cost_func:
        :param heuristic_func:
        :param available_action_function:
        :param rand_dev_prob: odd of randomly choosing an action versus taking the best action
        :return:
        """
        frontier_node = (tuple(self.start_pos), ())
        # open the most promising node from nodes_cost_dict
        # todo extend this code to sort, and then run parallel computation on each of the top-n promising nodes
        #todo change the dict to be a sorted list and make it faster for extraction
        while True:
            curr_pos = np.array(frontier_node[0])
            curr_plan = frontier_node[1]
            if cost_func(self.start_state_array,curr_plan) > self.max_cost:
                break
            if np.equal(curr_pos,self.goal_pos).all():  # we need a separate function since we do wildcard matching pass
                return curr_plan
            #end if
            # what actions are available
            avail_actions = available_action_function(self,self.start_state_array,curr_pos)
            possible_next_nodes = []
            for single_action in avail_actions:
                new_pos = self.transform_function(self.start_state_array,curr_pos,single_action)
                new_plan = list(curr_plan) + [single_action]
                new_heuristic = heuristic_func(self,new_pos)
                possible_next_nodes.append((new_pos,new_plan,new_heuristic))
            # end for loop
            #now with rand_dev_prob we take a random action, or we take the optimal action
            if random.uniform(0,1) < rand_dev_prob:
                next_node= random.choice(possible_next_nodes)
                frontier_node = (next_node[0],next_node[1])
            else:
                # in the early parts of the plan, since the heuristic is manhattan distance, it can go right or up.
                #however, because of the order of the nodes in possible next nodes, the min function chooses the
                # "go right" node initially, and preferentially. This causes a bias in plans. So we shuffle.
                random.shuffle(possible_next_nodes)
                min_cost_node = min(possible_next_nodes, key=lambda x: x[2])
                frontier_node = (min_cost_node[0],min_cost_node[1])
        #end while
        #if we reached here no plan
        return []
    #=======================================================================================

    def convert_actionSeq_to_StateSeq(self, plan_by_actionSeq):
        """
        :summary : convert a plan that is a sequence of tuples, that has the x, y position, into a series
        of states in dictionary format. We get the additional state features from our map of the gridworld
        :param plan_by_posSequence:
        :return:
        """
        xformed_plan = []
        curr_pos = np.array([0,0])
        for single_step in plan_by_actionSeq:
            #start with the curr_pos, then add the step and get the next curr_pos
            new_state = {}
            new_state["x"] = curr_pos[0]
            new_state["y"] = curr_pos[1]
            feature_vector = self.start_state_array[curr_pos[0]][curr_pos[1]]
            for idx in range(len(feature_vector)):
                if feature_vector[idx] != 0:
                    new_state[self.index_to_feature_dict[idx]] = feature_vector[idx]
            #end inner for
            xformed_plan.append(new_state)
            curr_pos += np.array(single_step)
        #end outer for
        return xformed_plan
    #=======================================================================================
    @staticmethod #redefined as a static method. Now can be called directly using Class Name; no object creation necessary
    def extract_feature_from_state_of_plan(state):
        """
        assumes each state has a single feature
        helper function that takes a plan and returns a list of feature sequences

        states with no features are characterized by the "empty" feature

        :param state: state of an arbitrary plan
        :return:
        """

        #todo if state contains multiple features, then it should be stated what features he likes, right now choosing randomly what he likes
        for single_key in state.keys():
            if not (single_key == "x" or single_key == "y" or single_key == ANNOTATION_KEYWORD or single_key == "initial_state" or single_key == "goal_state"):
                # so the key represents a feature
                return single_key
        return "empty"

    # ================================================================================
    @staticmethod  # redefined as a static method. Now can be called directly using Class Name; no object creation necessary
    def check_state_equality(state_a, state_b):
        if state_a["x"] == state_b["x"] and state_a["y"] == state_b["y"]:
            return True
        return False
    #=======================================================
    def convert_plans_to_state_seqs_and_features(self,all_plans):
        """

        Summary : Converts plans from action sequences to state sequences. This assumes the initial state (position) is 0,0
        In parallel, all the L1 (single state features) and L2 features (ordered pairs of state features ) are also recorded.
        All features are binary for now.
        :param all_plans:
        :return converted plans:
        :return all_L1_features:
        :return all_L2_features:
        """


        converted_plans = []
        all_L1_features = set()
        all_L2_features = set()
        for single_plan in all_plans:
            #convert plan to state sequence
            state_seq = self.convert_actionSeq_to_StateSeq(single_plan)
            #--- extract the features from the state sequence.
            L1_features_seen = set()
            L2_features_seen = set()
            for single_state in state_seq:
                curr_vars = set(single_state.keys())
                curr_vars.remove("x")
                curr_vars.remove("y")
                #save the ordered pairwise product of the lists of prior features and curr features.
                #here product is [1,2] x [3,4] = [(1,3),(1,4),(2,3),(2,4)].
                L2_features_seen = L2_features_seen.union(set(itertools.product(L1_features_seen,curr_vars)))
                L1_features_seen = L1_features_seen.union(curr_vars)
            #end for loop through the state sequences
            converted_plans.append([state_seq,{"unknown": L1_features_seen.union(L2_features_seen) ,"like": None ,"dislike": None}])
            all_L1_features = all_L1_features.union(L1_features_seen)
            all_L2_features = all_L2_features.union(L2_features_seen)
        #end for loop through all plans
        return (converted_plans,all_L1_features,all_L2_features)
    #=======================================================

    def convert_plans_to_all_inclusive_format(self, plan_set):
        """
        Just like convert_plans_to_state_seqs_and_features function but works for any level-feature. Also the output
        plans are formatted like this:
        <state_sequence, {"unknown": sorted list of all features }>
        :param plan_set:
        :return:
        """