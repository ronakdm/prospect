import numpy as np
import torch

raw_dir = "{your directory here}"
data_dir = "data/iwildcam"
try:
    z = np.load(f"{raw_dir}/train_features.npy")
    y = np.load(f"{raw_dir}/train_labels.npy")
    zv = np.load(f"{raw_dir}/val_features.npy")
    yv = np.load(f"{raw_dir}/val_labels.npy")
    zt = np.load(f"{raw_dir}/test_features.npy")
    yt = np.load(f"{raw_dir}/test_labels.npy")
except FileNotFoundError:
    raise FileNotFoundError(
        f"This script assumes access to ResNet50 features on top of the 2020 iWildCam dataset to exist in {raw_dir}! Either place them in this directory, or use the already processed features in '/data/iwildcam'."
    )

# find the most frequently occurring classes in train
uy, c = np.unique(y, return_counts=True)
argsort = np.argsort(c)[::-1]
uy, c = uy[argsort], c[argsort]
uy, c = uy[1:], c[1:]  # leave out the background class

# Only keep classes with at least 100 training labels
num_classes_to_keep = (c >= 100).sum()  # = 60
classes_to_keep = uy[c >= 100]
remap_classes = {a: i for (i, a) in enumerate(classes_to_keep)}

idxs = np.isin(y, classes_to_keep)  # train elements
z = z[idxs]
y = np.array([remap_classes[a] for a in y[idxs]])

idxs = np.isin(yv, classes_to_keep)  # val elements
zv = zv[idxs]
yv = np.array([remap_classes[a] for a in yv[idxs]])

idxs = np.isin(yt, classes_to_keep)  # test elements
zt = zt[idxs]
yt = np.array([remap_classes[a] for a in yt[idxs]])


# Now sample from these arrays
def mysample(y, min_per_class, max_per_class, total, seed=0):
    rng = np.random.RandomState(seed)
    rand_idxs = rng.permutation(y.shape[0])  # Order to consider examples in
    out1 = []
    out2 = []
    counts = {a: 0 for a in np.unique(y)}  # keep counts of the classes we sample
    for i in rand_idxs:
        c = y[i]  # sampled class
        if counts[c] < min_per_class:
            out1.append(i)  # definitely keep this class
            counts[c] += 1
        elif counts[c] < max_per_class:
            out2.append(i)  # maybe keep this
            counts[c] += 1
    if len(out1) >= total:
        return np.array(out1)
    n_left = total - len(out1)  # number of elements we still want to sample
    out3 = rng.choice(out2, size=n_left, replace=False).tolist()
    return np.array(out1 + out3)


idxs = mysample(y, 100, 1000, 20000, seed=0)
z1 = z[idxs]
y1 = y[idxs]
torch.save(z1, f"new/train_embeddings.pt")
np.save(f"new/y_train.npy", y1)

idxs = mysample(yv, 50, 500, 5000, seed=1)
z1 = zv[idxs]
y1 = yv[idxs]
torch.save(z1, f"new/validation_embeddings.pt")
np.save(f"new/y_validation.npy", y1)


idxs = mysample(yt, 50, 500, 5000, seed=2)
z1 = zt[idxs]
y1 = yt[idxs]
torch.save(z1, f"new/test_embeddings.pt")
np.save(f"new/y_test.npy", y1)


# Run PCA:
z = torch.load(f"new/train_embeddings.pt")
zv = torch.load(f"new/validation_embeddings.pt")
zt = torch.load(f"new/test_embeddings.pt")

from sklearn.decomposition import PCA

pca = PCA(n_components=None)
pca.fit(z)
idxs = np.cumsum(pca.explained_variance_ratio_) < 0.99
print(idxs.sum() + 1)  # number of dimensions to keep

cutoff = (np.cumsum(pca.explained_variance_ratio_) < 0.99).sum() + 1

z1 = pca.transform(z)[:, :cutoff]
z1v = pca.transform(zv)[:, :cutoff]
z1t = pca.transform(zt)[:, :cutoff]

# normalize each datapoint to norm 1
z4 = z1 / np.linalg.norm(z1, axis=1)[:, None]
z4v = z1v / np.linalg.norm(z1v, axis=1)[:, None]
z4t = z1t / np.linalg.norm(z1t, axis=1)[:, None]

# np.save(f"new/X_train_2.npy", z4)
# np.save(f"new/X_validation_2.npy", z4v)
# np.save(f"new/X_test_2.npy", z4t)
