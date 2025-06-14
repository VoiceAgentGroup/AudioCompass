class BaseBenchmark:
    def __init__(self, **kwargs):
        pass

    def generate(self, model):
        """
        Generate results using the provided model.

        Args:
            model: The model to be used for generating results.

        Returns:
            list: A list of generated results.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def evaluate(self, results):
        """
        Evaluate the generated results.

        Args:
            results (list): The generated results to be evaluated.

        Returns:
            dict: The evaluation results.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

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
            dict: The evaluation results.
        """
        generated_results = self.generate(model)
        self.save_generated_results(generated_results, output_dir, model.model_name)
        evaluated_result = self.evaluate(generated_results)
        return evaluated_result