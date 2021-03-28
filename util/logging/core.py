import torch


class LogEntry(object):

    def __init__(self, init_losses=[], init_metrics=[]):
        assert isinstance(init_losses, list)
        self._losses = {}
        for key in init_losses:
            self._losses[key] = 0

        assert isinstance(init_metrics, list)
        self._metrics = {}
        for key in init_metrics:
            self._metrics[key] = 0
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        loss_str = ''
        for key, value in self._losses.items():
            if value != 0:
                loss_str += ' | {}: {:.4f}'.format(key, value)
        loss_str += ' | {}: {:.4f}'.format('Total', sum(self._losses.values()))

        metric_str = ''
        for key, value in self._metrics.items():
            if value != 0:
                metric_str += ' | {}: {:.4f}'.format(key, value)

        out = ''
        if len(loss_str) > 0:
            out += 'Losses\t{}\n'.format(loss_str)
        if len(metric_str) > 0:
            out += 'Metrics\t{}\n'.format(metric_str)

        if len(out) > 0:
            out = out[:-1]

        return out

    @property
    def losses(self):
        return self._losses
    
    @property
    def metrics(self):
        return self._metrics

    def clear(self):
        self._losses = {}
        self._metrics = {}

    def reset(self):
        for key in self._losses:
            self._losses[key] = 0
        for key in self._metrics:
            self._metrics[key] = 0

    def add_loss(self, key):
        if key not in self._losses:
            self._losses[key] = 0

    def add_metric(self, key):
        if key not in self._metrics:
            self._metrics[key] = 0

    def itemize(self):
        for key, value in self._losses.items():
            if isinstance(value, torch.Tensor):
                self._losses[key] = value.item()

        for key, value in self._metrics.items():
            if isinstance(value, torch.Tensor):
                self._metrics[key] = value.item()

    def absorb(self, other_log):
        assert isinstance(other_log, LogEntry)

        for key, value in other_log.losses.items():
            if key in self._losses:
                self._losses[key] += value
            else:
                self._losses[key] = value

        for key, value in other_log.metrics.items():
            if key in self._metrics:
                self._metrics[key] += value
            else:
                self._metrics[key] = value

    def average(self, N):
        for key in self._losses:
            self._losses[key] /= N

        for key in self._metrics:
            self._metrics[key] /= N

    def to_dict(self):
        return { 'losses' : self._losses, 'metrics' : self.metrics }
