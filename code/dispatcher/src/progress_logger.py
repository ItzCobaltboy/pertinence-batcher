"""
A pymoo Callback that fires after every generation — logs progress and
saves a checkpoint every few generations, so a long unattended run can be
checked on (or resumed from) mid-flight.
"""

import os
import numpy as np
from pymoo.core.callback import Callback

import constants as c


class ProgressLogger(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.generation = 0

    def notify(self, algorithm):
        self.generation += 1

        objectives = algorithm.pop.get("F")   # (pop_size, 2): [alpha_sys_loss, avg_flops_G]
        chromosomes = algorithm.pop.get("X")

        best_alpha_sys = 100 * (1 - objectives[:, 0].min())
        self.logger.info(
            f"-- Generation {self.generation}/{c.GENERATIONS} done  |  "
            f"best_alpha_sys={best_alpha_sys:.2f}%  "
            f"best_low_flops={objectives[:,1].min():.3f}G --"
        )

        if self.generation % c.CHECKPOINT_EVERY_N_GENERATIONS == 0 or self.generation == c.GENERATIONS:
            self._save_checkpoint(chromosomes, objectives)

    def _save_checkpoint(self, chromosomes, objectives):
        os.makedirs(c.NSGA2_DIR, exist_ok=True)
        path = os.path.join(c.NSGA2_DIR, f"checkpoint_gen{self.generation:03d}.npz")
        np.savez(path, population=chromosomes, objectives=objectives, generation=self.generation)
        self.logger.info(f"Checkpoint -> {path}")
