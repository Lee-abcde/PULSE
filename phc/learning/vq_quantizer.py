import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.uniform_(-1.0 / 2, 1.0 / 2)
        # self.embedding.weight.data.uniform_(-1.0 / 256, 1.0 / 256)
        # self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim = -1, keepdim=True)  # project to sphere
        # self.embedding.weight.data[:] *= 10
        

    def forward(self, z, return_perplexity=False, return_loss = True):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        # records = [
        #     (240, 0.2923)
        # ]
        # self.visualize_vq(records=records)
        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        if return_loss:
            loss = torch.mean((z_q - z.detach())**2) + self.beta * torch.mean((z_q.detach() - z)**2)
            # loss =  self.beta * torch.mean((z_q.detach() - z)**2)

            # preserve gradients
            z_q = z + (z_q - z).detach()
        else:
            loss = torch.tensor(0.0).to(z.device)
        
        if return_perplexity:
            min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype) # measuring utilization
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
            return loss, z_q, min_encoding_indices, perplexity
        else:
            return loss, z_q, min_encoding_indices

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def visualize_vq(self, n_angles=100, device='cpu', method='pca', perplexity=30, records=None):
        """
        可视化 quantizer embedding 的相位轨迹。
        支持 PCA / t-SNE 降维，并可选择高亮记录点。

        Args:
            n_angles (int): 每个 embedding 轨迹采样角度数
            device (str): 运行设备 ('cpu' or 'cuda:0')
            method (str): 'pca' 或 'tsne'
            perplexity (int): t-SNE 参数
            records (list[tuple] or None): [(state_idx, phase_value)], 如果提供则高亮这些点
        """
        embeddings = self.embedding.weight.data.to(device)  # (n_e, e_dim)
        n_e, e_dim = embeddings.shape

        # 生成均匀角度 [-pi, pi]
        angles = torch.linspace(-np.pi, np.pi, n_angles, device=device)
        angles = angles.unsqueeze(0).repeat(n_e, 1)  # (n_e, n_angles)

        # 圆上采样点
        y0 = torch.cos(angles)
        y1 = torch.sin(angles)
        y = torch.stack((y0, y1), dim=-1)  # (n_e, n_angles, 2)

        # reshape embedding
        d = e_dim // 2
        state = embeddings.reshape(n_e, 1, d, 2)  # (n_e, 1, d, 2)
        y = y.unsqueeze(2)  # (n_e, n_angles, 1, 2)
        points = torch.matmul(state, y.transpose(-1, -2))  # (n_e, n_angles, d, 1)
        points = points.squeeze(-1).reshape(n_e, n_angles, -1)  # (n_e, n_angles, d)

        # flatten for dimensionality reduction
        all_points = points.reshape(n_e * n_angles, -1).cpu().numpy()

        # ----------- 降维部分 -----------
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(all_points)
            title = "VQ Embeddings Phase Circle (PCA)"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto')
            reduced = reducer.fit_transform(all_points)
            title = f"VQ Embeddings Phase Circle (t-SNE, perplexity={perplexity})"
        else:
            raise ValueError("method must be either 'pca' or 'tsne'")

        reduced = reduced.reshape(n_e, n_angles, 2)

        # ----------- 绘图部分 -----------
        plt.figure(figsize=(7, 7))

        if records is None:
            # 没有记录 -> 画全部轨迹
            cmap = plt.cm.get_cmap('hsv', n_e)
            for i in range(n_e):
                plt.plot(reduced[i, :, 0], reduced[i, :, 1], alpha=0.6)
                plt.scatter(reduced[i, 0, 0], reduced[i, 0, 1], c='k', s=10)  # 起点黑点
        else:
            # 有记录 -> 画灰色轨迹 + 红点
            for i in range(n_e):
                plt.plot(reduced[i, :, 0], reduced[i, :, 1], alpha=0.2, color='gray')

            for state_idx, phase_val in records:
                angle_idx = int(phase_val * (n_angles - 1))
                x, y_ = reduced[state_idx, angle_idx, :]
                plt.scatter(x, y_, color='red', s=40)

        plt.axis('equal')
        plt.grid(True)
        plt.title(title)
        plt.show()

        import ipdb;
        ipdb.set_trace()

class VectorQuantizer(nn.Module):
    def __init__(self, num_embed, embed_dim, beta, distance='l2',
                 anchor='probrandom', first_batch=False, contras_loss=False, n_dataset=1,
                 multiple_updater=1):
        """
            Taken from https://github.com/lyndonzheng/CVQ-VAE
            This class implements a feature buffer that stores previously encoded features

            This buffer enables us to initialize the codebook using a history of generated features
            rather than the ones produced by the latest encoders
        """
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False
        self.multiple_updater = multiple_updater

        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.usage = np.zeros((n_dataset, self.num_embed), dtype=np.int32)
        self.calling_from = 0
        if self.multiple_updater:
            updater = [self.UpdateModule(self.num_embed, self.decay, anchor) for _ in range(n_dataset)]
        else:
            updater = [self.UpdateModule(self.num_embed, self.decay, anchor)] * n_dataset
        self.updater = nn.ModuleList(updater)

    def get_weight(self):
        return self.embedding.weight

    class UpdateModule(nn.Module):
        def __init__(self, num_embed, decay, anchor):
            super().__init__()
            self.buffer = {}
            self.num_embed = num_embed
            self.decay = decay
            self.register_buffer("embed_prob", torch.zeros(self.num_embed))
            self.clear_buffer()
            self.anchor = anchor

        def clear_buffer(self):
            self.buffer['encodings'] = []
            self.buffer['d'] = []
            self.buffer['z_flattened'] = []

        def update_prob(self, prob):
            self.embed_prob.mul_(self.decay).add_(prob, alpha=1 - self.decay)

        def get_alpha(self):
            return torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(1)

        def unpack_buffer(self, clear=True):
            encodings = torch.concat(self.buffer['encodings'], axis=0)
            d = torch.concat(self.buffer['d'], axis=0)
            z_flattened = torch.concat(self.buffer['z_flattened'], axis=0)
            if clear:
                self.clear_buffer()
            return encodings, d, z_flattened

        def update_buffer(self, encodings, d, z_flattened):
            self.buffer['encodings'].append(encodings)
            self.buffer['d'].append(d)
            self.buffer['z_flattened'].append(z_flattened)

        def get_update(self, embeddings):
            encodings, d, z_flattened = self.unpack_buffer(True)
            avg_probs = torch.mean(encodings, dim=0)
            self.update_prob(avg_probs)
            random_feat = self.sample_feat(d, z_flattened)
            decay = self.get_alpha()
            update = (1 - decay) * embeddings + decay * random_feat
            return update

        def sample_feat(self, d, z_flattened):
            if self.anchor == 'closest':
                sort_distance, indices = d.sort(dim=0)
                random_feat = z_flattened.detach()[indices[-1, :]]
            # feature pool based random sampling
            elif self.anchor == 'random':
                random_feat = self.pool.query(z_flattened.detach())
            # probability based random sampling
            elif self.anchor == 'probrandom':
                norm_distance = F.softmax(d.t(), dim=1)
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                random_feat = z_flattened.detach()[prob]
            return random_feat

    def set_caller(self, idx):
        self.calling_from = idx

    def clear_buffer(self):
        for updater in self.updater:
            updater.clear_buffer()

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, freeze_codebook=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_shape = z.shape
        z_flattened = z.view(-1, self.embed_dim)
        # records = [
        #     (156, 0.3390),
        # ]
        # self.visualize_vq(records=records)
        # self.visualize_vq(n_angles=30, method='tsne', records=records)
        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.matmul(z_flattened.detach(), self.embedding.weight.t())
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.matmul(normed_z_flattened, normed_codebook.t())

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]

        # quantise and unflatten
        z_q = self.embedding.weight[encoding_indices]
        # reshape back to match original input shape
        z_q = z_q.reshape(z_shape)

        if self.training:
            # compute loss for embedding
            commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            if freeze_codebook:
                codebook_loss = torch.tensor(0., device=z.device)
            else:
                codebook_loss = torch.mean((z_q - z.detach()) ** 2)
            loss = commitment_loss + codebook_loss
            # preserve gradients
            z_q = z + (z_q - z).detach()

            encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # min_encodings = encodings
        else:
            loss = torch.tensor(0., device=z.device)
            perplexity = torch.zeros(1, device=z.device)
            # min_encodings = torch.zeros(1, device=z.device)

        # update the running usage
        if self.training and self.calling_from >= 0 and not freeze_codebook:
            np.add.at(self.usage[self.calling_from], encoding_indices.detach().cpu().numpy(), 1)
            self.updater[self.calling_from].update_buffer(encodings, d, z_flattened)

        # contrastive loss
        if self.training and self.contras_loss and not freeze_codebook:
            sort_distance, indices = d.sort(dim=0)
            dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0,
                                                                                                   keepdim=True)
            dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
            dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
            contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
            loss = loss + contra_loss

        return loss, z_q, encoding_indices, perplexity

    def reinitialize(self):
        # online clustered reinitialisation for unoptimized points
        if self.training:
            if self.multiple_updater:
                updates = [self.updater[i].get_update(self.embedding.weight) for i in range(len(self.updater))]
                self.embedding.weight.data = torch.stack(updates, dim=0).mean(dim=0)
            else:
                updater = self.updater[0]
                self.embedding.weight.data = updater.get_update(self.embedding.weight)

    def visualize_vq(self, n_angles=100, device='cpu', method='pca', perplexity=30, records=None):
        """
        可视化 quantizer embedding 的相位轨迹。
        支持 PCA / t-SNE 降维，并可选择高亮记录点。

        Args:
            n_angles (int): 每个 embedding 轨迹采样角度数
            device (str): 运行设备 ('cpu' or 'cuda:0')
            method (str): 'pca' 或 'tsne'
            perplexity (int): t-SNE 参数
            records (list[tuple] or None): [(state_idx, phase_value)], 如果提供则高亮这些点
        """
        embeddings = self.embedding.weight.data.to(device)  # (n_e, e_dim)
        n_e, e_dim = embeddings.shape

        # 生成均匀角度 [-pi, pi]
        angles = torch.linspace(-np.pi, np.pi, n_angles, device=device)
        angles = angles.unsqueeze(0).repeat(n_e, 1)  # (n_e, n_angles)

        # 圆上采样点
        y0 = torch.cos(angles)
        y1 = torch.sin(angles)
        y = torch.stack((y0, y1), dim=-1)  # (n_e, n_angles, 2)

        # reshape embedding
        d = e_dim // 2
        state = embeddings.reshape(n_e, 1, d, 2)  # (n_e, 1, d, 2)
        y = y.unsqueeze(2)  # (n_e, n_angles, 1, 2)
        points = torch.matmul(state, y.transpose(-1, -2))  # (n_e, n_angles, d, 1)
        points = points.squeeze(-1).reshape(n_e, n_angles, -1)  # (n_e, n_angles, d)

        # flatten for dimensionality reduction
        all_points = points.reshape(n_e * n_angles, -1).cpu().numpy()

        # ----------- 降维部分 -----------
        if method == 'pca':
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(all_points)
            title = "VQ Embeddings Phase Circle (PCA)"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto')
            reduced = reducer.fit_transform(all_points)
            title = f"VQ Embeddings Phase Circle (t-SNE, perplexity={perplexity})"
        else:
            raise ValueError("method must be either 'pca' or 'tsne'")

        reduced = reduced.reshape(n_e, n_angles, 2)

        # ----------- 绘图部分 -----------
        plt.figure(figsize=(16, 8))  # Increase height slightly for the colorbar

        # --- Subplot 1: Phase Circle Trajectories ---
        ax1 = plt.subplot(1, 2, 1)

        # Draw faint background trajectories for all codes
        # This helps see the manifold structure
        for i in range(n_e):
            ax1.plot(reduced[i, :, 0], reduced[i, :, 1], alpha=0.1, color='gray', linewidth=0.5)

        if records:
            # Highlight specific records and connect them with lines
            record_points = []
            for code_idx, phase_val in records:
                if 0 <= code_idx < n_e:
                    # Logic: phase is [-0.5, 0.5], mapping to index [0, n_angles-1]
                    # Normalize phase from [-0.5, 0.5] to [0, 1]
                    norm_phase = phase_val + 0.5
                    # Map to index
                    angle_idx = int(norm_phase * (n_angles - 1))
                    # Clip to ensure bounds safety
                    angle_idx = max(0, min(n_angles - 1, angle_idx))

                    x, y_ = reduced[code_idx, angle_idx, :]
                    record_points.append((x, y_))

                    # Plot the point (keep the red dots for emphasis)
                    ax1.scatter(x, y_, color='red', s=30, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10)

            # Draw lines connecting the records with a color gradient
            if len(record_points) > 1:
                # Use 'viridis_r' colormap: goes from lighter yellow (fair) to darker blue/purple (dark)
                cmap = plt.cm.viridis_r
                num_segments = len(record_points) - 1

                for i in range(num_segments):
                    p1 = record_points[i]
                    p2 = record_points[i + 1]

                    # Calculate color based on progress through the records
                    progress = i / max(1, num_segments - 1)
                    color = cmap(progress)

                    # Draw the line segment
                    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, alpha=0.8, zorder=9)

                # Add a colorbar to indicate the temporal order
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(records) - 1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax1, ticks=[0, len(records) - 1], orientation='horizontal', fraction=0.04,
                                    pad=0.1)
                cbar.set_ticklabels(['Start', 'End'])
                cbar.set_label('Record Sequence (Fair -> Dark Color)')

        ax1.set_title(f"VQ Phase Trajectories \n(Red dots = usage records, Lines = sequence)")
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: Frequency Bar Chart ---
        plt.subplot(1, 2, 2)

        if records:
            # Count frequency of each code index
            indices = [r[0] for r in records]
            counts = Counter(indices)

            # Prepare x (all code indices) and y (counts)
            x_axis = range(n_e)
            y_axis = [counts.get(i, 0) for i in x_axis]

            # Color bars: Highlight used codes in blue, unused in light gray
            colors = ['steelblue' if c > 0 else 'lightgray' for c in y_axis]

            plt.bar(x_axis, y_axis, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

            # Annotate Top 5 codes
            if indices:
                top_k = 5
                most_common = counts.most_common(top_k)
                info_text = "Top Used Codes:\n" + "\n".join([f"Idx {code}: {cnt}x" for code, cnt in most_common])

                # Place text box in top right
                plt.text(0.95, 0.95, info_text,
                         transform=plt.gca().transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.title(f"Code Usage Histogram (Total: {len(records)})")
            plt.xlabel("Codebook Index")
            plt.ylabel("Frequency")
            plt.xlim(-1, n_e)

            # Dynamic Y-limit to make it look nice
            if len(indices) > 0:
                plt.ylim(0, max(y_axis) * 1.15)

        else:
            plt.text(0.5, 0.5, "No Records Provided", ha='center', va='center')

        plt.tight_layout()
        plt.show()

        import ipdb;
        ipdb.set_trace()


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super(EmbeddingEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim) 
        
        # weight = weight/weight.norm(dim = -1, keepdim=True) # project to sphere
        
        self.weight = nn.Parameter(weight, requires_grad=False)
        # self.weight.data.uniform_(-1.0 / num_tokens, 1.0 / num_tokens)
        self.weight.data.uniform_(-1.0, 1.0)
        
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False) # counts for how many times the code is used.
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_emb_avg):
        self.update_idxes = new_emb_avg.abs().sum(dim = -1) > 0
        self.embed_avg.data[self.update_idxes] = self.embed_avg.data[self.update_idxes].mul_(self.decay).add(new_emb_avg[self.update_idxes], alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + num_tokens*self.eps) * n)
        embed_normalized = self.embed_avg 
        embed_normalized[self.update_idxes] = self.embed_avg[self.update_idxes] / smoothed_cluster_size.unsqueeze(1)[self.update_idxes]
        self.weight.data.copy_(embed_normalized)
        
        
        

class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5):
        super(EMAVectorQuantizer, self).__init__()

        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z, return_perplexity=False):
        z_flattened = z.view(-1, self.codebook_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        min_encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
        
        if self.training and self.embedding.update:
            encoding_sum = min_encodings.sum(0)
            embed_sum = min_encodings.transpose(0, 1) @ z_flattened
            
            self.embedding.cluster_size_ema_update(encoding_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()
        
        if return_perplexity:
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
            return loss, z_q, min_encoding_indices, perplexity
        else:
            return loss, z_q, min_encoding_indices

