from torch import nn


def parse_optimizer(optimizer_str: str, **kwargs):
    if "Adam" in optimizer_str:
        return {
            "name": "adam",
            "params": kwargs,
        }
    elif "SGD" in optimizer_str:
        return {
            "name": "sgd",
            "params": kwargs,
        }
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_str}")


def decorate_optimizer(optimizer):
    def decorated_optimizer(lr: float, weight_decay: float, model: nn.Module, **kwargs):
        return optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)

    return decorated_optimizer
