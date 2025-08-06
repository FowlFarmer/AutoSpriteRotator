import matplotlib.pyplot as plt

class LiveMetricPlotter:
    def __init__(self, tracked_keys):
        """
        Initialize the live plotter.

        Args:
            tracked_keys (list of str): Metrics to track and plot.
        """
        plt.ion()
        self.tracked_keys = tracked_keys
        self.data = {k: [] for k in tracked_keys}
        self.max_vals = {k: 1e-8 for k in tracked_keys}  # avoid div by zero
        self.batches = []

        self.fig, self.ax = plt.subplots()
        self.lines = {
            k: self.ax.plot([], [], label=k)[0]
            for k in tracked_keys
        }

        self.ax.set_xlabel("Batch")
        self.ax.set_ylabel("Scaled Value (0–1)")
        self.ax.set_title("Live Training Metrics")
        self.ax.legend()
        self.ax.grid(True)

    def update(self, batch, metric_dict):
        """
        Update the plot with new metric values.

        Args:
            batch (int): Current batch number.
            metric_dict (dict): Dict of new metric values for tracked keys.
        """
        self.batches.append(batch)

        for k in self.tracked_keys:
            val = metric_dict.get(k, None)
            if val is not None:
                self.data[k].append(val)
                self.max_vals[k] = max(self.max_vals[k], val)

                # Scale data
                scaled = [v / self.max_vals[k] for v in self.data[k]]
                self.lines[k].set_xdata(self.batches)
                self.lines[k].set_ydata(scaled)

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
