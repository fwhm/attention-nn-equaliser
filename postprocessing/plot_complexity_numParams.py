import matplotlib.pyplot as plt
import numpy as np
import inspect
import os
from utils.utils import check_makedir

# For BRNN32
import matplotlib.pyplot as plt
import numpy as np

save = True
labels = ['BiLSTM', 'BiGRU', 'BiGRU_trim', 'part_BiGRU', 'part_BiGRU_trim']
RMpS = [390976, 296512, 292160, 175744, 171392]
RMpS = np.asarray(RMpS)
params = [14978, 12546, 8194, 10370, 8194]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, RMpS/20, width, label='MpS')
ax.bar(labels, params, width, bottom=RMpS/20,
       label='trainable params')

# ax.set_ylabel('Scores')
ax.set_title('Complexity comparison', fontsize=16)
plt.rcParams['font.size'] = '16'
# ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=12)

if save:
    postprocessing_path = os.getcwd()
    path_to_save = os.path.join(postprocessing_path, "plots", 'complexity')
    check_makedir(path_to_save)
    absfilename = os.path.join(path_to_save, "complexity_comparison.svg")
    plt.savefig(absfilename, format='svg')

plt.show()


# labels = ['RMpS', 'No. trainable params']
# labels = ['RMpS']
# bilstm = [390976, 14978]
# bigru = [296512, 12546]
# bigru_trim = [292160, 8194]
# part_bigru = [175744, 10370]
# part_bigru_trim = [171392, 8194]
# x = np.arange(len(labels))  # the label locations
# x = np.array([0, 0.4])
# width = 0.06 # the width of the bars
# x=np.arange(1)
# fig, ax = plt.subplots()
#
# rects1 = ax.bar(x - 2*width, bilstm[0], width, label='bilstm')
# rects2 = ax.bar(x - width, bigru[0], width, label='bigru')
# rects3 = ax.bar(x, bigru_trim[0], width, label='bigru_trim')
# rects4 = ax.bar(x + width, part_bigru[0], width, label='part_bigru')
# rects5 = ax.bar(x + 2*width, part_bigru_trim[0], width, label='part_bigru_trim')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('#number of')
# ax.set_title('complexity and #params')
# ax.set_xticks(x, labels)
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)
# ax.bar_label(rects4, padding=3)
# ax.bar_label(rects5, padding=3)
#
# # fig.tight_layout()
#
# plt.show()