
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    pass


def metric_average(val, name):
    """Average a given quantity over all processes

    Arguments:
        val {torch.tensor} -- data to average
        name {str} -- name of the data

    Returns:
        torch.tensor -- Averaged quantity
    """
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()
