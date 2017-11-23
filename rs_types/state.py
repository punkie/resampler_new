class BasicState:
    def __init__(self,
                 dataset=None,
                 output_dir=None,
                 sampling_algorithm=None,
                 normal_classify_thread_finished=None,
                 resample_classify_thread_finished=None,
                 classified_data_normal_case=None,
                 classified_data_resampled_case=None,
                 classification_algorithm=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.sampling_algorithm = sampling_algorithm
        self.normal_classify_thread_finished = normal_classify_thread_finished
        self.resample_classify_thread_finished = resample_classify_thread_finished
        self.classified_data_normal_case = classified_data_normal_case
        self.classified_data_resampled_case = classified_data_resampled_case
        self.classification_algoritm = classification_algorithm