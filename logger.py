from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Tensorboard logger backed by torch.utils.tensorboard (no TensorFlow
    dependency needed)."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar value at a given step."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()
