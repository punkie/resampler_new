from imblearn.over_sampling import RandomOverSampler
from general_functions import write_dataset_to_csv
from imblearn.datasets import fetch_datasets

def do_random_oversampling(dataset, output_directory):
    #dataset = fetch_datasets()['ecoli']
    #y_values  = dataset['target']
    #x_values = dataset['data']
    y_values = dataset['y_values']
    x_values = dataset['x_values']
    ros = RandomOverSampler()
    x_resampled_values, y_resampled_values = ros.fit_sample(x_values, y_values)
    write_dataset_to_csv(output_directory + "/test.txt", x_values_param=x_resampled_values, y_values_param=y_resampled_values)

#def extract_dataset_info()