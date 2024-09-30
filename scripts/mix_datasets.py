import h5py
import numpy as np

# _f1 = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/experiments/RLHF_test/oracle/new_dataset.hdf5"
_f2 = "/home/local/USHERBROOKE/levj1404/Documents/TrackToLearn/data/experiments/RLHF_test/oracle/new_dataset.hdf5"
# _f2 = "/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/ismrm_test2.hdf5"
_new = "/home/local/USHERBROOKE/levj1404/Documents/TractOracleNet/TractOracleNet/datasets/ismrm2015_1mm/ismrm_test2_combined_new_ds.hdf5"

# f1 = h5py.File(_f1, 'r')
f2 = h5py.File(_f2, 'r')
f3 = h5py.File(_new, 'a')

data2 = np.array(f2['streamlines/data'])
scores2 = np.array(f2['streamlines/scores'])

# Copy data from f1
# f1.copy('streamlines', f3)
# f3.attrs['nb_points'] = f1.attrs['nb_points']
# f3.attrs['version'] = f1.attrs['version']

# Get the number of streamlines total
len1 = len(f3['streamlines/scores'])
len2 = len(f2['streamlines/scores'])
nb_streamlines = len1 + len2

# Resize the dataset
f3['streamlines/data'].resize(nb_streamlines, axis=0)
f3['streamlines/scores'].resize(nb_streamlines, axis=0)

# Setup indices
indices = np.arange(len1, nb_streamlines)
np.random.shuffle(indices)

ps_indices = np.random.choice(len2, len2, replace=False)
idx = indices[:len2]

# Add data from f2
for i, st, sc in zip(idx, data2, scores2):
    f3['streamlines/data'][i] = st
    f3['streamlines/scores'][i] = sc

# f1.close()
f2.close()
f3.close()

