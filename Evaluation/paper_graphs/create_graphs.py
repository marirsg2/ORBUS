import numpy as np
from matplotlib import pyplot as plt
import re
import pprint as pp
# define all the global variables here
class create_graphs:
    def __init__(self):
        self.preference_distribution = "power_law"
        self.noise_settings = [0.3, 0.6, 0.9]
        self.preference_probability_cases = [(0.1, 0.1),
                                        # (0.1, 0.2),
                                        # (0.1, 0.4),
                                        (0.05, 0.25),
                                        ]
        self.no_of_case_per_settings = 5
        self.no_of_exp_per_settings = 2
        self.number_of_rounds = 3
        self.prefix = "test"
        self.suffix = ".result"
        self.all_file_names = []
        self.create_all_file_names()

    def create_all_file_names(self):
        for single_noise in self.noise_settings:
            for single_pref_prob in self.preference_probability_cases:
                exp_count = 0
                for exp_number in range(self.no_of_exp_per_settings):
                    for case_number in range(self.no_of_case_per_settings):
                        file_name = self.prefix + \
                                    "_noise" + \
                                    str(single_noise) + \
                                    "_" + \
                                    str(single_pref_prob[0]) + \
                                    str(single_pref_prob[1]) + \
                                    "_" + \
                                    self.preference_distribution + \
                                    "_" + \
                                    str(exp_count) + \
                                    self.suffix
                        exp_count += 1
                        self.all_file_names.append(file_name)

    def create_all_graphs(self):
        for single_file_name in self.all_file_names:
            open_file = "./data/" + single_file_name
            with open(open_file, 'r') as exp_file:
                # find the bayes error list and rbus error list
                file_content = exp_file.read()
                bayes_error = re.search(r'ROUND  '+ str(self.number_of_rounds) + '(.*)BAYES MODEL target and prediction(.*?)]', file_content, re.DOTALL).group(2)
                mle_error = re.search(r'ROUND  '+ str(self.number_of_rounds) + '(.*)LINEAR MODEL target and prediction(.*?)]', file_content, re.DOTALL).group(2)

                bayes_error = eval(bayes_error + ']')
                mle_error = eval(mle_error + ']')

                self.make_graph(bayes_error, mle_error, single_file_name)

    def create_all_line_graphs(self):
        t1 = []
        t2 = []

        for i in range(len(self.all_file_names) + 1):
            if i % 5 == 0 and i != 0:
                t1.append(t2)
                t2 = []
            t2.append(i)

        t3 = []
        for i in range(len(t1)):
            if i % 2 == 0:
               t3.append(t1[i])

        folder = './line_graphs/'
        for file_set_indices in t3:
            all_line_data = []
            for file_idx in file_set_indices:
                open_file = './line_graph_data/' + self.all_file_names[file_idx]
                with open(open_file, 'r') as exp_file:
                    file_content = exp_file.read()
                    bayes_error = re.search(r'ROUND  '+ str(self.number_of_rounds) + '(.*)BAYES ERROR LIST (.*)BAYES ERROR LIST(.*)]', file_content, re.DOTALL).group(2)
                    bayes_error_1 = np.array(eval(bayes_error[:bayes_error.index('M')]))

                open_file = './line_graph_data/' + self.all_file_names[file_idx + 5]
                with open(open_file, 'r') as exp_file:
                    file_content = exp_file.read()
                    bayes_error = re.search(
                        r'ROUND  ' + str(self.number_of_rounds) + '(.*)BAYES ERROR LIST (.*)BAYES ERROR LIST(.*)]',
                        file_content, re.DOTALL).group(2)
                    bayes_error_2 = np.array(eval(bayes_error[:bayes_error.index('M')]))

                all_line_data.append((bayes_error_1 + bayes_error_2) / 2)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            labels = ['random_sampling', 'variance', 'variance_with_feature', 'rbus', 'rbus_with_feature']
            ax1.set_xticks ([1,2,3,4])
            for idx, single_line in enumerate(all_line_data):
                ax1.plot([1,2,3,4], single_line, label=labels[idx])

            ax1.legend(loc=3)
            file_name = self.all_file_names[file_idx]
            save_file_name = file_name[:file_name.index('.result')] + ".png"
            plt.savefig(folder + save_file_name)
            plt.close()

    def create_all_bayes_cases(self):
        t1 = []
        t2 = []

        for i in range(len(self.all_file_names) + 1):
            if i % 5 == 0 and i != 0:
                t1.append(t2)
                t2 = []
            t2.append(i)

        folder = './graphs/'
        for file_set_indices in t1:
            bayes_errors_list = []
            max_values = []
            for file_idx in file_set_indices:
                open_file = "./data/" + self.all_file_names[file_idx]
                with open(open_file, 'r') as exp_file:
                    # find the bayes error list
                    file_content = exp_file.read()
                    bayes = re.search(
                        r'ROUND  ' + str(self.number_of_rounds) + '(.*)BAYES MODEL target and prediction(.*)BAYES MODEL target and prediction(.*)]' ,
                        file_content, re.DOTALL).group(2)
                    bayes = bayes[:bayes.index(']')+1]
                    bayes = eval(bayes)
                    bayes = np.array(bayes)
                    bayes_errors = np.abs(bayes[:, 0] - bayes[:, 2])
                    max_values.append(np.max(bayes_errors))
                    bayes_errors_list.append([bayes[:, 0], bayes_errors, file_idx])

            y_max = np.max(max_values)
            for single_bayes_error in bayes_errors_list:
                plt.figure()
                plt.ylim([0, y_max])
                x, y = (single_bayes_error[0], single_bayes_error[1])
                plt.scatter(x, y, c='tab:blue', label='bayes', alpha=0.3, edgecolors='none')
                file_name = self.all_file_names[single_bayes_error[2]]
                save_file_name = file_name[:file_name.index('.result')] + ".png"
                plt.savefig(folder + "bayes_" + save_file_name)
                plt.close()

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
    exp.create_all_line_graphs()