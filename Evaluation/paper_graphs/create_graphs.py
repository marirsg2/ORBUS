import numpy as np
from matplotlib import pyplot as plt
import re
import pprint as pp

# define all the global variables here
# feature feedback, random_sampling_enabled, include_discovery, include_gain, include_feature_distinguishing, include_prob_term
# it shouldn't exceed 10
graph_settings = [
    [1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 1],
]

file_name = "RBUS_output_results_12_09AM_on_November_12__2019 .txt"

case_settings = [
    "feature feedback",
    "random_sampling_enabled",
    "include_discovery",
    "include_gain",
    "feature_distinguishing",
    "include_prob_term"
]


class create_graphs:
    def __init__(self):
        # gather the data for all the cases
        with open(file_name, 'r') as data_file:
            data = data_file.read()
            cases = re.findall(r'CASE DESCRIPTIONS((.|\n)+?)==', data)
            for single_case in cases:
                # extract case settings
                single_case = single_case[0]
                for single_case_setting in case_settings:
                    truth_value = re.findall(r"%s\s*=\s*\w+" % single_case_setting, single_case)[0]
                    if truth_value.find("True") != -1:
                        print("true")
                    elif truth_value.find("False") != -1:
                        print("false")
                    else:
                        print("unknown truth value")
        # create graph for all the cases


    def make_graph(self, bayes, mle, file_name):
        save_file_name = file_name[:file_name.index('.result')] + ".png"
        folder = './graphs/'
        bayes = np.array(bayes)
        mle = np.array(mle)

        bayes_errors = np.abs(bayes[:, 0] - bayes[:, 2])
        mle_errors = np.abs(mle[:, 0] - mle[:, 1])

        plt.figure()
        plt.ylim([0, np.max([np.max(bayes_errors), np.max(mle_errors)])])
        x, y = (bayes[:, 0], bayes_errors)
        plt.scatter(x, y, c='tab:blue', label='bayes', alpha=0.3, edgecolors='none')
        plt.savefig(folder + "bayes_" + save_file_name)

        plt.figure()
        plt.ylim([0, np.max([np.max(bayes_errors), np.max(mle_errors)])])
        x, y = (mle[:, 0], mle_errors)
        plt.scatter(x, y, c='tab:red', label='mle', alpha=0.3, edgecolors='none')
        plt.savefig(folder + "mle_" + save_file_name)

if __name__ == "__main__":
    print("Printing Graphs for the paper, main and supplemental section")
    exp = create_graphs()