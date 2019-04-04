# %%


from sklearn import datasets

digits = datasets.load_digits()

X = digits.data[:500]
y = digits.target[:500]

# %%

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# %%

target_ids = range(len(digits.target_names))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 7))

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
