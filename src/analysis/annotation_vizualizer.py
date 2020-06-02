"""
Create histograms of different subsets of Aff-Wild2 dataset.
By choosing 'balancing_mode' we can visualize whole dataset or downsampled subset.
By choosing 'train_test' we can visualize train or test samples of the dataset.
"""
from pylab import *

from src.data import DataSet

dataset = DataSet()
# valences,arousals = dataset.get_val_ar(balancing_mode='balanced',max_mode='mean')
valences, arousals = dataset.get_val_ar(balancing_mode='all', train_test='test')
# valences,arousals = dataset.get_val_ar(mode='neg_ar_pos_val')
print(len(valences))
print(len(arousals))
res_hist = hist2d(valences, arousals, bins=40, cmap=cm.jet)
density = res_hist[0] / len(valences)
s = np.sum(density)
colorbar().ax.tick_params(axis='y', direction='out')
# savefig("/Users/denisrangulov/Google Drive/EmotionRecognition/figures/b_mean_train_frames.png", bbox_inches='tight')
# savefig("/Users/denisrangulov/Google Drive/EmotionRecognition/figures/train_neg_400_train_frames.png", bbox_inches='tight')
# savefig("/Users/denisrangulov/Google Drive/EmotionRecognition/figures/train_neg_ar_pos_val_400_train_frames.png", bbox_inches='tight')
savefig("/Users/denisrangulov/Google Drive/EmotionRecognition/figures/all_test_frames.png", bbox_inches='tight')
