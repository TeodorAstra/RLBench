from wandb.integration.sb3 import WandbCallback
import wandb

class CustomWandbCallback(WandbCallback):
    def __init__(
        self,
        success_rate_window: int = 1000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.success_rate_window = success_rate_window
        self.successes_window = 0
        self.total_episodes_window = 0
        self.success_rates = []

    def _on_step(self) -> bool:
        # Call the original _on_step method
        result = super()._on_step()

        # Calculate success rate
        if self.locals.get("episode"):
            self.total_episodes_window += 1
            if self.locals["info"].get("is_success"):
                self.successes_window += 1

        # Log success rate if necessary
        if self.total_episodes_window >= self.success_rate_window:
            self.log_success_rate()

        return result

    def _on_training_end(self) -> None:
        super()._on_training_end()
        # Log final success rate
        self.log_success_rate()

    def log_success_rate(self) -> None:
        success_rate = (
            self.successes_window / self.total_episodes_window
        ) if self.total_episodes_window > 0 else 0
        self.success_rates.append(success_rate)
        # If the window size exceeds the specified window, remove the oldest success rate
        if len(self.success_rates) > self.success_rate_window:
            self.success_rates.pop(0)
        # Log the average success rate of the last 'success_rate_window' episodes
        wandb.log({"success_rate_last_1000_episodes": sum(self.success_rates) / len(self.success_rates)})
