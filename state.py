class BasicState:
    def __init__(self, dataset=None, output_dir=None,
                 sampling_algorithm=None, resampled_dataset=None,
                 normal_classify_thread_finished=None,
                 resample_classify_thread_finished=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.sampling_algorithm = sampling_algorithm
        self.normal_classify_thread_finished = normal_classify_thread_finished
        self.resample_classify_thread_finished = resample_classify_thread_finished