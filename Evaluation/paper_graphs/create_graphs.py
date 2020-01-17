import numpy as np
from matplotlib import pyplot as plt
import re
import pprint as pp
import copy
import glob
# define all the global variables here
# feature feedback, random_sampling_enabled, include_discovery, include_gain, include_feature_distinguishing, include_prob_term
# required_graph_settings = [
#     # [1, 0, 1, 1, 1, 1],
#     # [1, 0, 1, 0, 1, 1],
#     [1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 0, 1],
#     [1, 1, 1, 1, 0, 1]
# ]

required_graph_settings = [
[1, 0, 1 ,1, 0 ,1] ,
[1, 0, 0 ,1, 0 ,1],
[1, 0, 1 ,1, 0 ,0],
[1, 0, 0 ,1, 0 ,0]
]

legends = copy.deepcopy(required_graph_settings)

color_array = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k',
]

file_name = "RBUS_output_results_01_52PM_on_November_13__2019.txt"

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
        self.settings_to_data = None

    def extract_relevant_data(self, file_name=file_name):
        self.graph_data = []
        # gather the data for all the cases
        with open(file_name, 'r') as data_file:
            data = data_file.read()
            cases = re.findall(r'CASE DESCRIPTIONS((.|\n)+?)==', data)
            for single_case in cases:
                # extract case settings
                single_case = single_case[0]
                # given_case_setting
                given_case_settings = []
                for single_case_setting in case_settings:
                    truth_value = re.findall(r"%s\s*=\s*\w+" % single_case_setting, single_case)[0]
                    if truth_value.find("True") != -1:
                        given_case_settings.append(1)
                    elif truth_value.find("False") != -1:
                        given_case_settings.append(0)
                    else:
                        print("unknown truth value")

                # print(given_case_settings)

                # now gather the data here.
                pre = "REGION BAYES ERROR LIST"
                post = "INTERESTING"
                case_data = eval(re.search('%s((.|\n)*)%s' % (pre, post), single_case).groups(1)[0])
                # print(case_data)

                if given_case_settings in required_graph_settings:
                    self.graph_data.append(case_data)
                    required_graph_settings.remove(given_case_settings)
                    # return case_data

    def create_graphs(self):
        # save_file_name = "basic.png"
        # folder = './graphs/'
        x = np.arange(len(self.graph_data[0]))

        plt.figure()
        for idx, data in enumerate(self.graph_data):
            plt.plot(x, data, color=color_array[idx])
            plt.legend(legends, loc='upper right')
        plt.show()
        # plt.savefig(folder + "mle_" + save_file_name)

    def create_graph_all_data(self):
        graph_data = []
        legends = []
        for k, v in self.settings_to_data.items():
            legends.append(k)
            graph_data.append(np.average(v, axis=0))

        x = np.arange(len(graph_data[0]))
        fig = plt.figure()
        plt.xlabel("round number")
        plt.ylabel("error")
        for idx, data in enumerate(graph_data):
            plt.plot(x, data, color=color_array[idx])
            plt.legend(legends, loc='upper right')
        plt.show()

    def create_all_the_data(self):
        setting_to_data = {str(i): [] for i in required_graph_settings}
        all_files = [file for file in glob.glob("*.txt")]
        for single_file in all_files:
            temp_req_settings = copy.deepcopy(required_graph_settings)
            with open(single_file, 'r') as data_file:
                data = data_file.read()
                cases = re.findall(r'CASE DESCRIPTIONS((.|\n)+?)==', data)
                for single_case in cases:
                    # extract case settings
                    single_case = single_case[0]
                    # given_case_setting
                    given_case_settings = []
                    for single_case_setting in case_settings:
                        truth_value = re.findall(r"%s\s*=\s*\w+" % single_case_setting, single_case)[0]
                        if truth_value.find("True") != -1:
                            given_case_settings.append(1)
                        elif truth_value.find("False") != -1:
                            given_case_settings.append(0)
                        else:
                            print("unknown truth value")

                    # print(given_case_settings)

                    # now gather the data here.
                    pre = "REGION BAYES ERROR LIST"
                    post = "INTERESTING"
                    case_data = eval(re.search('%s((.|\n)*)%s' % (pre, post), single_case).groups(1)[0])
                    # print(case_data)

                    if given_case_settings in temp_req_settings:
                        setting_to_data[str(given_case_settings)].append(case_data)
        self.settings_to_data = setting_to_data


if __name__ == "__main__":
    print("Printing Graphs for the paper, main and supplemental section")
    exp = create_graphs()
    exp.create_all_the_data()
    exp.create_graph_all_data()
