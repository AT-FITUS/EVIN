import os
from datetime import datetime
from cornac.experiment import Experiment as CornacExperiment


class Experiment(CornacExperiment):
    def __init__(
        self,
        eval_method,
        models,
        metrics,
        user_based=True,
        show_validation=True,
        verbose=False,
        save_dir=None,
    ):
        super().__init__(
            eval_method,
            models,
            metrics,
            user_based=user_based,
            show_validation=show_validation,
            verbose=verbose,
            save_dir=save_dir,
        )
        self.output_file = None

    def run(self):
        """Run the Cornac experiment"""
        self._create_result()

        # overwrite verbosity setting of evaluation method and models
        # if Experiment verbose is True
        if self.verbose:
            self.eval_method.verbose = self.verbose
            for model in self.models:
                model.verbose = self.verbose

        for model in self.models:
            test_result, val_result = self.eval_method.evaluate(
                model=model,
                metrics=self.metrics,
                user_based=self.user_based,
                show_validation=self.show_validation,
            )

            self.result.append(test_result)
            if self.val_result is not None:
                self.val_result.append(val_result)

            if self.save_dir and (not isinstance(self.result, CVExperimentResult)):
                model.save(self.save_dir)

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        output += "\nTEST:\n...\n{}".format(self.result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        self.output_file = os.path.join(save_dir, "CornacExp-{}.log".format(timestamp))
        with open(self.output_file, "w") as f:
            f.write(output)
