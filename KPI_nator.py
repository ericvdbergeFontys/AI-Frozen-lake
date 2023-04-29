import matplotlib.pyplot as plt
import seaborn as sns

class KPInator:
    def __init__(self):
        """Creates graphs to analyse our agent"""

    def visualize_value_evaluation(self, value_evaluation, shape):
        sns.heatmap(value_evaluation.reshape(shape),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False,
                     fmt="3g")
        plt.show()