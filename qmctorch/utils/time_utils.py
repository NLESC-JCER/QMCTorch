from time import time


def timeit(method):
    def timed(*args, **kwargs):
        t0 = time()
        result = method(*args, **kwargs)
        t1 = time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = (t1-t0)
        else:
            print('%s %f s' % (method.__name__, (t1-t0)))
        return result

    return timed


def timeline(cmd, name=''):
    t0 = time()
    out = eval(cmd)
    t1 = time()
    print(name, t1-t0)
    return out
