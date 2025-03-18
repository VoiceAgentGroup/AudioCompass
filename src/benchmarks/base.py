class BaseBenchmark:
    def __init__(self):
        pass

    def run(self, model, output_dir):
        """
        Run the benchmark with the given model and output directory.

        Args:
            model: The model to be benchmarked.
            output_dir (str): The directory where the output will be saved.

        Returns:
            None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")