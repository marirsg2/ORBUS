import matplotlib.pyplot as plt

class GenerateVisualizations:

    @staticmethod
    def plot_histogram(data, bins=30):
        plt.hist(data, bins=bins, facecolor="green", alpha=0.75)
        plt.show()