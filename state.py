class BasicState:
    def __init__(self, dataset=None, output_dir=None, sampling_algorithm=None, resampled_dataset=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.sampling_algorithm = sampling_algorithm