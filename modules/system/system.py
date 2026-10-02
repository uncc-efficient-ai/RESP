"""Single-device pruning; generation and evaluation use Accelerate workers."""

world_size = 1
rank = 0
ddp = False
device = 'cpu'


def init_system(c):
    global device
    device = c.system.device


def setup_model(model, data, tokenizer, c):
    return model.to(device)
