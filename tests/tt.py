import h5py
filename_hdf= '/data/viktor/CP1_P16_530hr_time_test_005nm_01/recon/CP1_P16_530hr_time_test_005nm_01_run20.cxi'
def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')

with h5py.File(filename_hdf, 'r') as hf:
    print(hf)
    h5_tree(hf)
