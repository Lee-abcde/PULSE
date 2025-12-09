import joblib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
import random

# -----------------------
# 1. Load Data (Same as before)
# -----------------------
pkl_path = "../../data/amass/text_embedding_dict_clip.pkl"
text_embed_dict = joblib.load(pkl_path)
all_keys = list(text_embed_dict.keys())

# Filter out 'transition'
texts = [t for t in all_keys if "transition" not in t.lower()]

# Convert to numpy
embeds = []
for t in texts:
    v = text_embed_dict[t]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    else:
        v = np.array(v)
    embeds.append(v)
embeds = np.vstack(embeds)

# -----------------------
# 2. PCA
# -----------------------
pca = PCA(n_components=2)
points = pca.fit_transform(embeds)

# -----------------------
# 3. Define Clusters (Keywords)
# -----------------------
# You can change these keywords to whatever motions you want to check
motion_keywords = ["walk", "run", "jump", "dance", "sit", "stand", "kick"]

# Helper to find category for a text
def get_category(text):
    t_lower = text.lower()
    for kw in motion_keywords:
        if kw in t_lower:
            return kw
    return "other" # Default if no keyword matches

# Assign a category to every point
categories = [get_category(t) for t in texts]
categories = np.array(categories)

# -----------------------
# 4. Plot by Category
# -----------------------
plt.figure(figsize=(14, 12))

# Define a color palette (using a distinct colormap)
unique_cats = ["other"] + motion_keywords
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))

# Plot "other" first so it stays in the background
other_mask = (categories == "other")
plt.scatter(points[other_mask, 0], points[other_mask, 1],
            c="lightgrey", label="other", s=15, alpha=0.3)

# Plot each specific motion cluster on top
for i, cat in enumerate(motion_keywords):
    mask = (categories == cat)
    if np.sum(mask) > 0: # Only plot if points exist
        plt.scatter(points[mask, 0], points[mask, 1],
                    label=cat, s=30, alpha=0.8)

# Add labels (Optional: Label a few random points from 'other' or specific clusters)
# Here we label a random subset of 100 points to keep it clean
indices_to_label = random.sample(range(len(texts)), 100)
texts_adjust = []
for i in indices_to_label:
    plt.text(points[i, 0], points[i, 1], texts[i], fontsize=8, alpha=0.7)

plt.legend(title="Motion Type", fontsize=12, markerscale=2)
plt.title(f"PCA of CLIP Embeddings (Clustered by Motion Keyword)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()