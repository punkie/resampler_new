import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from imblearn.datasets import fetch_datasets


def load_and_print_ds():
    ts = fetch_datasets()['thyroid_sick']
    print(ts.data.shape)
    target_classes = sorted(Counter(ts.target).items())
    print(target_classes)

    labels = ['']
    healty = [target_classes[0][1]]
    sick = [target_classes[1][1]]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, healty, width, label='Здрави')
    rects2 = ax.bar(x + width / 2, sick, width, label='Болни')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Брой примери')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    load_and_print_ds()