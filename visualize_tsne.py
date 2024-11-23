import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用绝对路径加载数据
file_path = '/lab/kirito/clip-fairness/checkpoints/biggan.npz'
data = np.load(file_path)
f_g = data['f_g'] #ours
f_g1 = data['f_g1']#clip
labels = data['labels']

# 随机选择1000个样本，确保f_g和f_g1选择的是相同的样本
num_samples = 1000
random_indices = np.random.choice(len(f_g), num_samples, replace=False)
f_g_sampled = f_g[random_indices]
f_g1_sampled = f_g1[random_indices]
labels_sampled = labels[random_indices]

# Apply t-SNE to reduce dimensionality to 2D for both f_g and f_g1
tsne = TSNE(n_components=2, random_state=0)
features_tsne_g = tsne.fit_transform(f_g_sampled)
features_tsne_g1 = tsne.fit_transform(f_g1_sampled)

# Plotting the t-SNE reduced features with labels for f_g
plt.figure(figsize=(10, 8))
plt.scatter(features_tsne_g[labels_sampled == 0, 0], features_tsne_g[labels_sampled == 0, 1], marker='o', color='b', label='real')
plt.scatter(features_tsne_g[labels_sampled == 1, 0], features_tsne_g[labels_sampled == 1, 1], marker='x', color='r', label='fake')
plt.title('Ours ')
plt.legend()
plt.show()
plt.savefig('clip.png')

# Plotting the t-SNE reduced features with labels for f_g1
plt.figure(figsize=(10, 8))
plt.scatter(features_tsne_g1[labels_sampled == 0, 0], features_tsne_g1[labels_sampled == 0, 1], marker='o', color='b', label='real')
plt.scatter(features_tsne_g1[labels_sampled == 1, 0], features_tsne_g1[labels_sampled == 1, 1], marker='x', color='r', label='fake')
plt.title('CLIP:ViT-B/16 ')
plt.legend()
plt.show()
plt.savefig('adapter.png')
