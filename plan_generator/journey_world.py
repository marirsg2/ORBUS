import random
from pprint import pprint

class Journey_World:
    def __init__(self, grid_size = (10, 10), number_of_features = 20, probability_of_feature = 0.3, seed=None):
        """
        :returns a grid represented by a list of dictionary. Each dict represents a cell in the grid. It would have
        the x, y coordinates(that start from (0, 0) to (grid_size-1)). Each cell can have one state-variable at max and it
        is given in the dict by its name as key and value as 1.

        :algorithm go over each cell and enable a feature there with probability probability_of_features. Select a
        feature from number_of_feature features. All L-1 feature or state-variables are [a-z]. So maximum of 26 are allowed.

        for handling duplicates: each feature has a equal chance of getting selected initially. Once a feature is picked,
        its probability of being picked is halved; e.g. [0.25 0.25 0.25 0.25]  -> [0.125 0.25 0.25 0.25]. The weights would be re-normalized
        to [0.14 0.28 0.28 0.28].

        :param grid_size: rows, columns
        :param number_of_features:
        :param probability_of_feature:
        :param seed: to make a fixed grid configuration
        """

        if number_of_features > 26:
            raise Exception("Journey world cannot have more than 26 L-1 features")

        if seed is not None:
            random.seed(seed)

        l1_features_list = [(index,  chr(feature)) for index, feature in enumerate(range(97, 97 + number_of_features))] # creates a list of L-1 features starting from 'a'
        l1_features_probability_weights = [1 for i in range(number_of_features)]

        grid_cells_list = []
        for x in range(1, grid_size[0] + 1):
            for y in range(1, grid_size[1] + 1):
                cell_dict = {"x" : x, "y" : y}
                random_number = random.random()

                if random_number <= probability_of_feature:
                    # an event with probability_of_feature has occurred. So place a feature in this cell
                    # pick a feature
                    (index, feature) = random.choices(l1_features_list, l1_features_probability_weights)[0]
                    l1_features_probability_weights[index] /= 2
                    cell_dict[feature] = 1 # turn that feature on for this cell
                grid_cells_list.append(cell_dict)

                if x == 1 and y == 1:
                    cell_dict["start"] = 1

                if x == grid_size[0] and y == grid_size[1]:
                    cell_dict["goal"] = 1

        self.grid_cells_list = grid_cells_list
        self.l1_features_list = [feature_tuple[1] for feature_tuple in l1_features_list]
        random.seed() # resetting seed in case it was set earlier

    def get_l1_features_in_grid(self):
        return self.l1_features_list

    def print_grid(self):
        pprint(self.grid_cells_list)

    def get_grid_cells_list(self):
        return self.grid_cells_list

if __name__ == "__main__":
    print("testing create grid")
    journey_world = Journey_World()
    journey_world.print_grid()