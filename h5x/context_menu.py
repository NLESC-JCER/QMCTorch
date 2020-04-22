from PyQt5 import QtWidgets
from h5xplorer.menu_tools import *
from h5xplorer.menu_plot import *
import numpy as np


def context_menu(self, treeview, position):
    """Generate a right-click menu for the items"""

    # make sure tha there is only one item selected
    all_item = get_current_item(self, treeview, single=False)

    if len(all_item) == 1:

        item = all_item[0]
        _context_menu_group(self, item, treeview, position)
        _context_menu_data(self, item, treeview, position)

    else:
        pass


def _context_menu_group(self, item, treeview, position):

    try:

        _type = self.root_item.data_file[item.name].attrs['type']

        # if _type == 'molecule':
        #     molgrp = self.root_item.data_file[item.name]
        #     _context_mol(item, treeview, position, molgrp)

        # if _type == 'single_point':
        #     _context_sparse(item, treeview, position)

        if _type == 'sampling_traj':
            _context_sampling_trajectory(self,
                                         item,
                                         treeview,
                                         position)

        if _type == 'opt':
            _context_optimization(
                self, item, treeview, position)

    except Exception as inst:
        print(type(inst))
        print(inst)
        return

# def _context_mol(item, treeview, position, molgrp):

#     menu = QtWidgets.QMenu()
#     actions = {}
#     list_operations = ['Load in PyMol', 'Load in VMD', 'PDB2SQL']

#     for operation in list_operations:
#         actions[operation] = menu.addAction(operation)
#     action = menu.exec_(treeview.viewport().mapToGlobal(position))

#     _, cplx_name, mol_name = item.name.split('/')
#     mol_name = mol_name.replace('-', '_')

#     if action == actions['Load in VMD']:
#         viztools.create3Ddata(mol_name, molgrp)
#         viztools.launchVMD(mol_name)

#     if action == actions['Load in PyMol']:
#         viztools.create3Ddata(mol_name, molgrp)
#         viztools.launchPyMol(mol_name)

#     if action == actions['PDB2SQL']:
#         db = pdb2sql(molgrp['complex'].value)
#         treeview.emitDict.emit({'sql_' + item.basename: db})


# def _context_single_point(item, treeview, position):
#     menu = QtWidgets.QMenu()
#     actions = {}
#     list_operations = ['Load in PyMol', 'Load in VMD', 'PDB2SQL']

def _context_sampling_trajectory(self, item, treeview, position):

    menu = QtWidgets.QMenu()
    actions = {}
    list_operations = ['Plot Energy Walkers', 'Plot Blocking']

    for operation in list_operations:
        actions[operation] = menu.addAction(operation)
    action = menu.exec_(treeview.viewport().mapToGlobal(position))

    if action == actions['Plot Energy Walkers']:

        grp = get_current_hdf5_group(self, item)
        data = grp['local_energy'][()]
        data_dict = {'_walkers_energy': data}
        treeview.emitDict.emit(data_dict)

        data_dict = {}
        data_dict['exec_cmd'] = "plot_walkers_traj(_walkers_energy)"
        treeview.emitDict.emit(data_dict)

    if action == actions['Plot Blocking']:

        grp = get_current_hdf5_group(self, item)
        data = grp['local_energy'][()]
        data_dict = {'_walkers_energy': data}
        treeview.emitDict.emit(data_dict)

        data_dict = {}
        data_dict['exec_cmd'] = "plot_block(_walkers_energy)"
        treeview.emitDict.emit(data_dict)


def _context_optimization(self, item, treeview, position):

    menu = QtWidgets.QMenu()
    actions = {}
    list_operations = ['Plot Energy']

    for operation in list_operations:
        actions[operation] = menu.addAction(operation)
    action = menu.exec_(treeview.viewport().mapToGlobal(position))

    if action == actions['Plot Energy']:

        grp = get_current_hdf5_group(self, item)
        data = grp['local_energy'][()]
        data_dict = {'_eloc': data}
        treeview.emitDict.emit(data_dict)

        data_dict = {}
        data_dict['exec_cmd'] = "plot_energy(_eloc)"
        treeview.emitDict.emit(data_dict)


def _context_menu_data(self, item, treeview, position):

    try:
        data = get_group_data(get_current_hdf5_group(self, item))

        if data is None:
            list_operations = ['Print attrs']

        elif data.ndim == 1:
            list_operations = ['Print attrs',
                               '-', 'Plot Hist', 'Plot Line']

        elif data.ndim == 2:
            list_operations = ['Print attrs',
                               '-', 'Plot Hist', 'Plot Map']

        else:
            list_operations = ['Print attrs']

        action, actions = get_actions(
            treeview, position, list_operations)

        if action == actions['Print attrs']:
            send_dict_to_console(self, item, treeview)

        if 'Plot Hist' in actions:
            if action == actions['Plot Hist']:
                plot_histogram(self, item, treeview)

        if 'Plot Line' in actions:
            if action == actions['Plot Line']:
                plot_line(self, item, treeview)

        if 'Plot Map' in actions:
            if action == actions['Plot Map']:
                plot2d(self, item, treeview)

    except Exception as inst:
        print(type(inst))
        print(inst)
        return
