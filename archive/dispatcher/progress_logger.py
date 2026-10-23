"""
A pymoo Callback that fires after every generation — logs progress and
saves a checkpoint every few generations, so a long unattended run can be
checked on (or resumed from) mid-flight.
"""

import os
import numpy as np
from pymoo.core.callback import Callback


class ProgressLogger(Callback):
    """Logs best alpha_sys / lowest model cost each generation; checkpoints
    the population every config.CHECKPOINT_EVERY_N_GENERATIONS generations."""

    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config
        self.generation = 0

    def notify(self, algorithm):
        """pymoo calls this once per completed generation."""
        self.generation += 1

        objectives = algorithm.pop.get("F")   # (pop_size, 2): [alpha_sys_loss, avg_model_cost]
        chromosomes = algorithm.pop.get("X")

        best_alpha_sys = 100 * (1 - objectives[:, 0].min())
        self.logger.info(
            f"-- Generation {self.generation}/{self.config.GENERATIONS} done  |  "
            f"best_alpha_sys={best_alpha_sys:.2f}%  "
            f"best_low_cost={objectives[:,1].min():.3f} {self.config.MODEL_COST_UNIT} --"
        )

        if (self.generation % self.config.CHECKPOINT_EVERY_N_GENERATIONS == 0
                or self.generation == self.config.GENERATIONS):
            self._save_checkpoint(chromosomes, objectives)

    def _save_checkpoint(self, chromosomes, objectives):
        """Writes the current population + objectives to
        results/nsga2/checkpoint_genNNN.npz."""
        os.makedirs(self.config.NSGA2_DIR, exist_ok=True)
        path = os.path.join(self.config.NSGA2_DIR, f"checkpoint_gen{self.generation:03d}.npz")
        np.savez(path, population=chromosomes, objectives=objectives, generation=self.generation)
        self.logger.info(f"Checkpoint -> {path}")
