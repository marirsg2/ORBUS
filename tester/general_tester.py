

import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import timedcall, metric
from scipy import stats

from manager.main_manager import Manager
from plan_generator.journey_world import Journey_World
from plan_generator.journey_world_planner import Journey_World_Planner

#---------------------------------

planner_pickle_file = "planner_pickle.p"
pool_plans_pickle_file = "pool_plans_pickle.p"
clustered_data_file = "clustered_data_file.p"
elbow_plot_data_file = "elbow_plot_data_file.p"
converted_pool_plans_pickle_file = "conv_pool_plans_pickle_file.p"
k_plot_data = "k_plot_data.p"

def test_structure():
    domain = Journey_World()
    planner = Journey_World_Planner(domain.get_grid_cells_list())
    with open(planner_pickle_file,"wb") as dest:
        pickle.dump(planner,dest)

def test_manager():
    manager = Manager(num_dataset_plans=100)
    manager.sample_by_DBS()

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

#====================================================
def test_plan_generation(num_plans= 1000):
    """
    :param num_plans:
    :return:
    """
    planner = None
    with open(planner_pickle_file,"rb") as src:
        planner = pickle.load(src)
    #end with
    all_plans = planner.get_quickly_n_plans(num_plans)
    with open(pool_plans_pickle_file,"wb") as dest:
        pickle.dump(all_plans,dest)

#====================================================
def test_plan_feature_extraction():
    """
    :summary : get the plans , ask to get the L1 and L2 features, and store in the intended format
    :return:
    """


    planner = None
    with open(planner_pickle_file, "rb") as src:
        planner = pickle.load(src)
    # end with
    all_plans = None
    with open(pool_plans_pickle_file, "rb") as src:
        all_plans = pickle.load( src)
    #now convert the plans to the format of <state sequences><features>
    (converted_plans,all_L1,all_L2) = planner.convert_plans_to_state_seqs_and_features(all_plans)
    with open(converted_pool_plans_pickle_file, "wb") as dest:
        pickle.dump(converted_plans, dest)
        pickle.dump(all_L1, dest)
        pickle.dump(all_L2, dest)

#====================================================
def test_encoding_and_Kmedoids(num_medoids = 6, tolerance = 0.25):
    """
    :return:
    """
    planner = None
    with open(planner_pickle_file, "rb") as src:

        planner = pickle.load(src)
    # end with
    (converted_plans,all_L1,all_L2) = (None,None,None)
    with open(converted_pool_plans_pickle_file, "rb") as src:
        converted_plans = pickle.load(src)
        all_L1 = pickle.load(src)
        all_L2 = pickle.load(src)
    #assign consistent indices to L1 and L2 features
    all_L2 = set([x[0]+x[1] for x in all_L2]) #makes string features (descriptions) from tuples of strings, helps sorting
    features_list_index_map = sorted(list(all_L1.union(all_L2)))
    dim_size = len(features_list_index_map)
    #now the index of a feature in features_list_index_map, is the dimension index
    encoded_plans = []
    for single_plan in converted_plans:
        plan_encoding = np.zeros(dim_size,dtype=float)
        on_indices = [features_list_index_map.index(x) for x in single_plan[1]["unknown"] if isinstance(x,str)]
        #if it is not a string, then it is a tuple.
        on_indices += [features_list_index_map.index("".join(x)) for x in single_plan[1]["unknown"] if not isinstance(x,str)]
        for x in on_indices:
            plan_encoding[x]=1.0
        #end for loop
        encoded_plans.append(plan_encoding)
    #end outer for loop
    #now run k_medoid clustering
    start_medoids = list(np.random.choice(np.array(range(len(encoded_plans))) ,size=num_medoids,replace= False))
    # start_medoids = [20,30,121] #these should be a random set of medoids. and repeated many times
    clusters,medoids = planTrace_clustering(start_medoids, encoded_plans, tolerance=tolerance, show=True)
    for single_cluster in clusters:
        print("===========================================")
        print (single_cluster)
    print(medoids)
    with open(clustered_data_file, "wb") as dest:
        state_seq_plans = converted_plans #this is just for readability.
        pickle.dump(state_seq_plans, dest)
        pickle.dump(encoded_plans, dest)
        pickle.dump(clusters, dest)
        pickle.dump(medoids, dest)
#====================================================

def test_clustering_quality(cross_cluster_compare = False):
    """

    :return:
    """

    with open(clustered_data_file, "rb") as src:
        state_seq_plans = pickle.load(src)
        encoded_plans = pickle.load(src)
        clusters = pickle.load(src)
        medoids = pickle.load(src)

    sum_manhattan_distances = 0.0
    #todo ALSO use .try connectivity radius
    #compare the distances , min, mean , max, within the same cluster
    for cluster_idx in range(len(clusters)):
        curr_cluster = clusters[cluster_idx]#these are just indices
        curr_medoid_idx = medoids[cluster_idx]
        curr_medoid = encoded_plans[curr_medoid_idx]
        #compute the distances from each of the points to the medoid and store it in vector
        distances = np.array([sum(abs(encoded_plans[x] - curr_medoid)) for x in curr_cluster])
        all_statistics = stats.describe(distances)
        sum_manhattan_distances += np.sum(distances)
        print(all_statistics)

    if cross_cluster_compare:
        print(all_statistics)
        print("================cross cluster comparison================")
        #NOW compare the distances across clusters
        for cluster_idx in range(len(clusters)):
            for med_idx in range(len(medoids)):
                print("cluster = ",cluster_idx, " medoid = ",med_idx)
                if cluster_idx == med_idx:
                    continue
                curr_cluster = clusters[cluster_idx]  # these are just indices
                curr_medoid_idx = medoids[med_idx]
                curr_medoid = encoded_plans[curr_medoid_idx]
                # compute the distances from each of the points to the medoid and store it in vector
                distances = np.array([sum(abs(encoded_plans[x] - curr_medoid)) for x in curr_cluster])
                print(stats.describe(distances))

    return sum_manhattan_distances, all_statistics

#====================================================
def test_multipleK_and_plot(start = 2, end = 10,jump=1,compute = True, tolerance=0.25):
    """

    :return:
    """
    x_data = []
    y_data = []
    if compute == True:
        L1_sumError = []

        for k in range(start,end,jump):
            print("----computing for k =",k)
            test_encoding_and_Kmedoids(num_medoids=k, tolerance=tolerance)
            # verify the clustering
            sum_manhattan_dist, curr_stats = test_clustering_quality(cross_cluster_compare=False)
            L1_sumError.append(sum_manhattan_dist)
        x_data = list(range(start,end,jump))
        y_data = L1_sumError

        with open(elbow_plot_data_file, "wb") as dest:
            pickle.dump(x_data, dest)
            pickle.dump(y_data, dest)
    else:
        with open(elbow_plot_data_file, "rb") as src:
            x_data  = pickle.load(src)
            y_data  = pickle.load(src)

    #plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data,y_data, color='tab:blue')
    plt.show()

#===========================================================================
def test_cluster_printouts(num_samples_per_cluster = 5):
    """
    :summary : For each cluster printout the medoid and random samples of count == num_samples_per_cluster
    :return:
    """


    with open(converted_pool_plans_pickle_file, "rb") as src:
        _ = pickle.load( src)
        all_L1 =pickle.load( src)
        all_L2 =pickle.load( src)
    #assign consistent indices to L1 and L2 features
    all_L2 = set([x[0]+x[1] for x in all_L2]) #makes string features (descriptions) from tuples of strings, helps sorting
    features_list_index_map = sorted(list(all_L1.union(all_L2)))
    dim_size = len(features_list_index_map)
    #---

    with open(clustered_data_file, "rb") as src:
        state_seq_plans = pickle.load(src)
        encoded_plans = pickle.load(src)
        clusters = pickle.load(src)
        medoids = pickle.load(src)

    for single_cluster_idx in range(len(clusters)):
        curr_cluster = clusters[single_cluster_idx]
        curr_medoid_idx = medoids[single_cluster_idx]
        print("======================================")
        print("Cluster size = ",len(curr_cluster))
        # zip and filter on value == 0.
        medoid_plan_desc = [x for x in zip(features_list_index_map,encoded_plans[curr_medoid_idx]) if x[1] != 0.0]
        medoid_plan_desc = sorted(medoid_plan_desc,key=lambda x:x[0])
        print("medoid = ", medoid_plan_desc)
        medoid_enc = encoded_plans[curr_medoid_idx]


        curr_sample_size = num_samples_per_cluster
        if num_samples_per_cluster > len(curr_cluster):
            curr_sample_size =  len(curr_cluster)
        random_plans_in_cluster = random.sample(curr_cluster,curr_sample_size)
        for single_sample_idx in random_plans_in_cluster:
            sample_enc = encoded_plans[single_sample_idx]
            sample_plan_desc = [x for x in zip(features_list_index_map, encoded_plans[single_sample_idx]) if x[1] != 0.0]
            sample_plan_desc = sorted(sample_plan_desc, key=lambda x: x[0])
            print("sample difference = ", np.sum(np.abs(np.array(medoid_enc) - np.array(sample_enc) ))
                  ,"sample = ", sample_plan_desc)
        print("======================================")


#====================================================
if __name__ == "__main__":
    # test_structure()
    # #get plans, 10k and save them
    # test_plan_generation(num_plans= 10000)
    # #convert plans to state sequences and features (l1, l2), and pickle plan_generator
    # test_plan_feature_extraction()
    #encode and cluster
    test_encoding_and_Kmedoids(num_medoids=30, tolerance = 0.25) #this will also store them
    test_clustering_quality() #verify the saved clusters quality
    # test_multipleK_and_plot(start=2,end=200,jump=10,compute=True,tolerance=0.25)

    # test_manager()
    #todo sample from clusters, and see plans.
    test_cluster_printouts()
