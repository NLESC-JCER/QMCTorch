import h5py
import numpy as np
import torch
import warnings
from types import SimpleNamespace


def print_insert_error(obj, obj_name):
    warnings.warn('Issue inserting %s of type %s' %
                  (obj_name, str(type(obj))))


def print_load_error(obj, parent_grp):
    warnings.warn('Issue loading %s' % parent_grp.name)


def load_from_hdf5(obj, fname, root_name):
    h5 = h5py.File(fname, 'r')

    if root_name is None:
        root_grp = h5
    else:
        root_grp = h5[root_name]

    load_object(obj, root_name)
    h5.close()


def load_object(obj, parent_grp):
    if type(parent_grp) is h5py._hl.group.Group:
        load_group(obj, parent_grp)
    elif type(parent_grp) is h5py._hl.group.Dataset:
        load_data(obj, parent_grp)


def load_group(obj, parent_grp):
    try:
        for child_name, child_grp in children(parent_grp):
            load_object(obj.__getattribute__(child_name), child_grp)
    except:
        print_load_error(obj, parent_grp)


def load_data(obj, dataset):
    try:
        name = dataset.name.split('/')[-1]
        vals = dataset[()]
        obj.__setattr__('name', val)
    except:
        print_load_error(obj, dataset)


def dump_to_hdf5(obj, fname, root_name=None):
    """Save the object in a hdf5 file."""

    h5 = h5py.File(fname, 'w')

    if root_name is None:
        root_name = obj.__class__.__name__

    insert_object(obj, h5, root_name)
    h5.close()


def insert_object(obj, parent_grp, obj_name):
    """recursively insert the object in the parent group."""

    if haschildren(obj):
        insert_group(obj, parent_grp, obj_name)
    else:
        insert_data(obj, parent_grp, obj_name)


def insert_group(obj, parent_grp, obj_name):
    """Insert an object wich children in new group in the parent group."""
    if obj_name.startswith('_'):
        return

    try:
        own_grp = parent_grp.create_group(obj_name)
        for child_name, child_obj in children(obj):
            insert_object(child_obj,  own_grp, child_name)
    except:
        print_insert_error(obj, obj_name)


def insert_data(obj, parent_grp, obj_name):
    """Insert an obj whitout children in the parent group."""
    try:
        lookup_insert = {list: insert_list,
                         tuple: insert_tuple,
                         np.ndarray: insert_numpy,
                         torch.Tensor: insert_torch_tensor,
                         torch.nn.parameter.Parameter: insert_torch_parameter,
                         torch.device: insert_none,
                         type(None): insert_none}

        insert_fn = lookup_insert[type(obj)]
    except KeyError:
        insert_fn = insert_default

    try:
        insert_fn(obj, parent_grp, obj_name)
    except:
        print_insert_error(obj, obj_name)


def insert_default(obj, parent_grp, obj_name):
    """Base insert as a hdf5 dataset."""
    try:
        parent_grp.create_dataset(obj_name, data=obj)
    except:
        print_insert_error(obj, obj_name)


def insert_list(obj, parent_grp, obj_name):
    """Insert a list as a dataset or datagroup."""
    try:
        parent_grp.create_dataset(obj_name, data=obj)
    except:
        for il, l in enumerate(obj):
            try:
                insert_object(l, parent_grp, obj_name+'_'+str(il))
            except:
                print_insert_error(obj, obj_name)


def insert_tuple(obj, parent_grp, obj_name):
    insert_list(list(obj), parent_grp, obj_name)


def insert_numpy(obj, parent_grp, obj_name):
    """Insert a numpy nd array."""
    if obj.dtype.str.startswith('<U'):
        obj = obj.astype('S')
    insert_default(obj, parent_grp, obj_name)


def insert_torch_tensor(obj, parent_grp, obj_name):
    """Insert a torch tensor array."""
    insert_numpy(obj.detach().numpy(), parent_grp, obj_name)


def insert_torch_parameter(obj, parent_grp, obj_name):
    """Insert the data of a torch parameter."""
    insert_torch_tensor(obj.data, parent_grp, obj_name)


def insert_none(obj, parent_grp, obj_name):
    """Insert a none."""
    return


def haschildren(obj):
    """Checks if the obj has children."""
    ommit_type = [torch.nn.parameter.Parameter, torch.Tensor]
    if type(obj) in ommit_type:
        return False
    else:
        return hasattr(obj, '__dict__') or hasattr(obj, 'keys')


def children(obj):
    """Returns the items of the object."""

    if hasattr(obj, '__dict__'):
        return obj.__dict__.items()

    elif hasattr(obj, 'keys'):
        return obj.items()
