import h5py


def dump_to_hdf5(obj, fname):
    """Save the object in a hdf5 file."""
    h5 = h5py.File(fname, 'w')
    insert_object(obj, h5)
    h5.close()


def insert_object(obj, parent_grp, obj_name=None):
    """recursively insert the object in the parent group."""

    if obj_name is None:
        obj_name = obj.__class__.__name__

    if haschildren(obj):
        try:
            own_grp = parent_grp.create_group(obj_name)
            for child_name, child_obj in children(obj):
                insert_object(child_obj,  own_grp,
                              obj_name=child_name)
        except:
            print(obj_name, type(obj), obj)

    else:
        try:
            parent_grp.create_dataset(obj_name, data=obj)
        except:
            print(obj_name, type(obj), obj)


def haschildren(obj):
    """Checks if the obj has children."""
    return hasattr(obj, '__dict__') or hasattr(obj, 'keys')


def children(obj):
    """Returns the items of the object."""

    if hasattr(obj, '__dict__'):
        return obj.__dict__.items()
    elif hasattr(obj, 'keys'):
        return obj.items()
