import wandb
import logging
import os
import sys
from typing import Optional, Dict, Any


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

logger = logging.getLogger(__name__)


class CustomWandbCallback(BaseCallback):
    """Callback for logging experiments to Weights and Biases.

    Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used.

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
        log: What to log. One of "gradients", "parameters", or "all".
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.gradient_save_freq = gradient_save_freq
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn(
                "`log` must be one of `None`, 'gradients', 'parameters', or 'all', "
                "falling back to 'all'"
            )
            log = "all"
        self.log = log
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.successes = 0
        self.step_counter = 0
        self.success_rate = 0

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
            )
        wandb.config.setdefaults(d)

    #Step function modified to keep track of successes
    def _on_step(self) -> bool:
        super()._on_step()
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        
        if self.locals['infos'][0]['is_success']: #is_success added to the dict in step in rlbench_env.py
            self.successes += 1
        
        self.step_counter +=1
        return True

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {self.path}")


    def _on_rollout_end(self) -> None:
        
        self._wandblog_success_rate()
        #self._update_success_rate()
        """
        info = self.locals['infos']
        success = info[0]['is_success']
        print(info)
        print(success)
        """
    #Calculates and logs the successrate
    def _wandblog_success_rate(self) -> None:
        if self.step_counter != 0:
        # Calculate success rate
            success_rate = self.successes / self.step_counter
        else:
        # Set success rate to zero if step_counter is zero
            success_rate = 0
        wandb.log({"Success_rate": success_rate})

        self.successes = 0
        self.step_counter = 0

    def _update_success_rate(self) -> None:
        if self.step_counter != 0:
        # Calculate success rate
            self.success_rate = self.successes / self.step_counter
        else:
        # Set success rate to zero if step_counter is zero
            self.success_rate = 0
       
        self.successes = 0
        self.step_counter = 0

