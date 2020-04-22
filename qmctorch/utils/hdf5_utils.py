import h5py
import numpy as np
import torch
import warnings
from types import SimpleNamespace


def print_insert_error(obj, obj_name):
    warnings.warn('Issue inserting data %s of type %s' %
                  (obj_name, str(type(obj))))


def print_insert_type_error(obj, obj_name):
    warnings.warn('Issue inserting type of data %s (%s)' %
                  (obj_name, str(type(obj))))


def print_load_error(grp):
    warnings.warn('Issue loading %s' % grp)


def load_from_hdf5(obj, fname, obj_name):
    """Load the content of an hdf5 file in an object.

    Arguments:
        obj {object} -- object where to load the data
        fname {str} -- name pf the hdf5 file
        obj_name {str} -- name of the root group in the hdf5
    """

    h5 = h5py.File(fname, 'r')
    root_grp = h5[obj_name]

    load_object(root_grp, obj, obj_name)
    h5.close()


def load_object(grp, parent_obj, grp_name):
    """Load object attribute from the hdf5 group/data

    Arguments:
        grp {hdf5 group} -- the current group in the hdf5 architecture
        parent_obj {object} -- parent object
        grp_name {str} -- name of the group
    """

    for child_grp_name, child_grp in grp.items():

        if isgroup(child_grp):
            load_group(child_grp, parent_obj, child_grp_name)
        else:
            load_data(child_grp, parent_obj, child_grp_name)


def load_group(grp, parent_obj, grp_name):
    """Load object attribute from the hdf5 group

    Arguments:
        grp {hdf5 group} -- the current group in the hdf5 architecture
        parent_obj {object} -- parent object
        grp_name {str} -- name of the group
    """
    try:
        if not hasattr(parent_obj, grp_name):
            parent_obj.__setattr__(
                grp_name, SimpleNamespace())
        load_object(grp,
                    parent_obj.__getattribute__(
                        grp_name),
                    grp_name)
    except:
        print_load_error(grp_name)


def load_data(grp, parent_obj, grp_name):
    """Load object attribute from the hdf5 data

    Arguments:
        grp {hdf5 group} -- the current group in the hdf5 architecture
        parent_obj {object} -- parent object
        grp_name {str} -- name of the group
    """
    try:
        parent_obj.__setattr__(grp_name, grp[()])
    except:
        print_load_error(grp_name)


def lookup_cast(ori_type, current_type):
    raise NotImplementedError(
        "cast the data to the type contained in .attrs['type']")


def isgroup(grp):
    """Check if current hdf5 group is a group

    Arguments:
        grp {hdf5 group} -- hdf5 group or dataset

    Returns:
        bool -- True if the group is a group
    """

    return type(grp) == h5py._hl.group.Group or type(grp) == h5py._hl.files.File


def dump_to_hdf5(obj, fname, root_name=None):
    """Dump the content of an object in a hdf5 file.

    Arguments:
        obj {object} -- object to dump
        fname {str} -- name of the hdf5

    Keyword Arguments:
        root_name {str} -- root group in the hdf5 file (default: {None})
    """

    h5 = h5py.File(fname, 'a')

    if root_name is None:
        root_name = obj.__class__.__name__

    insert_object(obj, h5, root_name)
    h5.close()


def insert_object(obj, parent_grp, obj_name):
    """Insert the content of the object in the hdf5 file

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """

    if haschildren(obj):
        insert_group(obj, parent_grp, obj_name)
    else:
        insert_data(obj, parent_grp, obj_name)


def insert_group(obj, parent_grp, obj_name):
    """Insert the content of the object in a hdf5 group

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    if obj_name.startswith('_'):
        return

    if obj_name not in parent_grp:

        try:
            own_grp = parent_grp.create_group(obj_name)
            # for child_name, child_obj in children(obj):
            #     insert_object(child_obj,  own_grp, child_name)
            for child_name in get_children_names(obj):

                child_obj = get_child_object(obj, child_name)
                insert_object(child_obj,  own_grp, child_name)

        except:
            print_insert_error(obj, obj_name)

    else:
        print(
            'object %s already exists, keeping existing version of the data' % obj_name)


def insert_data(obj, parent_grp, obj_name):
    """Insert the content of the object in a hdf5 dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """

    if obj_name.startswith('_'):
        return

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
        # insert_type(obj, parent_grp, obj_name)
    except:
        print_insert_error(obj, obj_name)


def insert_type(obj, parent_grp, obj_name):
    """Insert the content of the type object in an attribute

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    try:
        parent_grp[obj_name].attrs['type'] = str(type(obj))
    except:
        print_insert_type_error(obj, obj_name)


def insert_default(obj, parent_grp, obj_name):
    """Default funtion to insert a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    try:
        parent_grp.create_dataset(obj_name, data=obj)
    except:
        print_insert_error(obj, obj_name)


def insert_list(obj, parent_grp, obj_name):
    """funtion to insert a list as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    try:
        parent_grp.create_dataset(obj_name, data=obj)
    except:
        for il, l in enumerate(obj):
            try:
                insert_object(l, parent_grp, obj_name+'_'+str(il))
            except:
                print_insert_error(obj, obj_name)


def insert_tuple(obj, parent_grp, obj_name):
    """funtion to insert a tuple as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    insert_list(list(obj), parent_grp, obj_name)


def insert_numpy(obj, parent_grp, obj_name):
    """funtion to insert a numpy array as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    if obj.dtype.str.startswith('<U'):
        obj = obj.astype('S')
    insert_default(obj, parent_grp, obj_name)


def insert_torch_tensor(obj, parent_grp, obj_name):
    """funtion to insert a torch tensor as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    insert_numpy(obj.detach().numpy(), parent_grp, obj_name)


def insert_torch_parameter(obj, parent_grp, obj_name):
    """funtion to insert a torch parameter as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    insert_torch_tensor(obj.data, parent_grp, obj_name)


def insert_none(obj, parent_grp, obj_name):
    """funtion to insert a None Type as a dataset

    Arguments:
        obj {object} -- object to save
        parent_grp {hdf5 group} -- group where to dump
        obj_name {str} -- name of the object
    """
    return


def haschildren(obj):
    """Check if the object has children

    Arguments:
        obj {object} -- the object to check

    Returns:
        bool -- True if the object has children
    """
    ommit_type = [torch.nn.parameter.Parameter, torch.Tensor]
    if type(obj) in ommit_type:
        return False
    else:
        return hasattr(obj, '__dict__') or hasattr(obj, 'keys')


def children(obj):
    """Returns the children of the object as items

    Arguments:
        obj {object} -- the object to check

    Returns:
        dict -- items 
    """

    if hasattr(obj, '__dict__'):
        return obj.__dict__.items()

    elif hasattr(obj, 'keys'):
        return obj.items()


def get_children_names(obj):
    """Returns the children names of the object as items

    Arguments:
        obj {object} -- the object to check

    Returns:
        dict -- items 
    """

    if hasattr(obj, '__dict__'):
        names = list(obj.__dict__.keys())

    elif hasattr(obj, 'keys'):
        names = list(obj.keys())

    if hasattr(obj, '__extra_attr__'):
        names += obj.__extra_attr__

    if hasattr(obj, 'state_dict'):
        names += list(obj.state_dict().keys())

    return list(set(names))


def get_child_object(obj, child_name):
    """Return the child object

    Arguments:
        obj {object} -- parent object
        child_name {str} -- cild name

    Returns:
        object -- child object
    """

    if hasattr(obj, '__getattr__'):
        try:
            return obj.__getattr__(child_name)

        except AttributeError:
            pass

    if hasattr(obj, '__getattribute__'):
        try:
            return obj.__getattribute__(child_name)

        except AttributeError:
            pass

    if hasattr(obj, '__getitem__'):
        try:
            return obj.__getitem__(child_name)

        except AttributeError:
            pass


def add_group_attr(filename, grp_name, attr):
    """Add attribute to a given group

    Arguments:
        filename {str} -- name of the file
        grp_name {str} -- name of the group
        attr {dict} -- attrivutes to add
    """

    h5 = h5py.File(filename, 'a')
    for k, v in attr.items():
        h5[grp_name].attrs[k] = v
    h5.close()


def register_extra_attributes(obj, attr_names):
    """Register extra attribute to be able to dump them

    Arguments:
        obj {object} -- the object where we want to add attr
        attr_names {list} -- a list of attr names
    """
    if not hasattr(obj, '__extra_attr__'):
        obj.__extra_attr__ = []
    obj.__extra_attr__ += attr_names
