class BaseBenchmark:
    def __init__(self):
        pass

    def save_generated_results(self, results, output_dir, model_name):
        """
        Save the generated results to a specified directory.

        Args:
            results (list): The generated results to be saved.
            output_dir (str): The directory where the results will be saved.
            model_name (str): The name of the model used for generating the results.

        Returns:
            None
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

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