import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

iris = sns.load_dataset("iris")
# sns.pairplot(iris, hue="species")
# plt.show()

pca = PCA(n_components=2)
# x_visible = pca.fit_transform(dataset['x_values'])
# y_values = dataset['y_values']
# negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
# f, ax = plt.subplots()
# f.set_size_inches(6, 6)
# ax.scatter(x_visible[y_values == negative_tc, 0], x_visible[y_values == negative_tc, 1], label="Negative class",
#            alpha=0.5, color='b')
# ax.scatter(x_visible[y_values == positive_tc, 0], x_visible[y_values == positive_tc, 1], label="Positive class",
#            alpha=0.5, color='r')
# if is_resampled_dataset:
#     ax.set_title("PCA Re-sampled data with {}".format(main_window.state.sampling_algorithm_data_tab.value[0]))
# else:
#     ax.set_title("PCA Original data")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlabel("PCA_X")
# ax.set_ylabel("PCA_Y")
# ax.legend(loc="upper left", prop={'size': 7})