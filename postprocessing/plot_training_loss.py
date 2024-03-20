# Internal
from global_config.global_config import OUTPUT_COLLECTION_PATH
from postprocessing.config.postprocessing_config import *

# External
import matplotlib.pyplot as plt
import numpy as np
import os

# CONFIG [start]
plch_str = '-1_Pdbm'
CASE = RNN_CASE
ARGS_STR = RNN_BILSTM_16
# CONFIG [end]
path_to_output = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR, plch_str)

training_loss_npy = os.path.join(path_to_output, 'training_loss.npy')
training_loss = np.load(training_loss_npy)
training_loss = np.asarray(training_loss, dtype=np.float64, order='C')
idx = np.arange(0, len(training_loss))
num_epoch_to_plot = 150
plt.plot(idx[:num_epoch_to_plot], training_loss[:num_epoch_to_plot])
plt.xlabel('num_epochs')
plt.ylabel('training loss')
plt.grid(linestyle='--')
plt.title('Training loss')
# plt.plot(idx, training_loss)
plt.show()
