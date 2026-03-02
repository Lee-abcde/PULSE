from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
from phc.learning.network_builder import init_mlp
import torch
import torch.nn as nn
import numpy as np
from phc.utils.torch_utils import project_to_norm
from phc.learning.vq_quantizer import EMAVectorQuantizer, Quantizer, VectorQuantizer
from phc.utils.flags import flags
from learning.vq_pae_modules import *
from functools import partial
DISC_LOGIT_INIT_SCALE = 1.0
import csv
import os
import matplotlib.pyplot as plt
from collections import deque
from sklearn.decomposition import PCA

class AMPZBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPZBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']

            self.proj_norm = self.task_obs_size_detail["proj_norm"]
            self.embedding_size = self.task_obs_size_detail['embedding_size']
            self.embedding_norm = self.task_obs_size_detail['embedding_norm']
            self.z_readout = self.task_obs_size_detail.get("z_readout", False)
            self.z_type = self.task_obs_size_detail.get("z_type", "sphere")
            self.dict_size = self.task_obs_size_detail.get("dict_size", 512)
            self.z_all = self.task_obs_size_detail.get("z_all", False)
            self.embedding_partion = self.task_obs_size_detail.get("embedding_partion", 1)
            # VQ-PAE
            self.window_size = kwargs['window_size']
            self.kinematic_obs_size = kwargs['kinematic_obs_size']
            self.prior_window_size = kwargs['prior_window_size']
            self.top_phase = 1.0
            self.bottom_phase = 0.0
            self.debug_index = 23
            self.debug_freq = 1.49
            self.debug_phase_p = torch.full((1, 1), self.bottom_phase, device='cuda')

            self.use_vae_prior = self.task_obs_size_detail.get("use_vae_prior", False)
            self.use_vae_fixed_prior = self.task_obs_size_detail.get("use_vae_fixed_prior", False)
            self.use_vae_clamped_prior = self.task_obs_size_detail.get("use_vae_clamped_prior", False)
            self.use_vae_sphere_prior = self.task_obs_size_detail.get("use_vae_sphere_prior", False)
            self.use_vae_sphere_posterior = self.task_obs_size_detail.get("use_vae_sphere_posterior", False)
            self.vae_prior_fixed_logvar = self.task_obs_size_detail.get("vae_prior_fixed_logvar", 0)
            self.vae_var_clamp_max = self.task_obs_size_detail.get("vae_var_clamp_max", 0)
            
            ##### Debug utils
            flags.idx = 0
            self.debug_idxes = [0] * self.embedding_partion
            
            if self.z_all:
                kwargs['input_shape'] = (self.embedding_size,)  # Task embedding size + self_obs
            else:
                kwargs['input_shape'] = (kwargs['self_obs_size'] + self.embedding_size,)  # Task embedding size + self_obs
                

            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

            self._build_z_mlp()
            if self.z_readout:
                self._build_z_reader()
            if self.separate:
                self._build_critic_z_mlp()
            self.input_text = "walk"
            # import os
            # import joblib
            # from pathlib import Path
            # script_dir = Path(__file__).parent
            # relative_pkl_path = "../../data/amass/text_embedding_dict_clip.pkl"
            # clip_pkl_path = script_dir / relative_pkl_path
            #
            # if os.path.exists(clip_pkl_path):
            #     print(f"Loading CLIP embeddings from: {clip_pkl_path} ...")
            #     with open(clip_pkl_path, "rb") as f:
            #         self.clip_embedding_dict = joblib.load(f)
            #
            #     print("Loaded keys:", list(self.clip_embedding_dict.keys())[:10])
            # else:
            #     print("File does not exist!")
            #
            # all_embeddings = []
            # all_texts = []
            #
            # for text, emb in self.clip_embedding_dict.items():
            #     emb = torch.as_tensor(emb).view(-1).cuda()
            #     all_embeddings.append(emb)
            #     all_texts.append(text)
            #
            # self.master_embeddings = torch.stack(all_embeddings)
            # self.master_texts = all_texts
            #
            # self.RECORD_FILE = clip_pkl_path.parent / 'vq_pae_records.csv'
            # self.initialize_record_file(self.RECORD_FILE)

            ###########################################
            # Visualization Normalized Observation
            ##########################################
            # self.vis_history_len = 300  # Keep last 300 frames
            # self.raw_data_buffer = deque(maxlen=self.vis_history_len)
            # self.pca_fitter = PCA(n_components=2)
            #
            # # Setup interactive plotting
            # plt.ion()
            # self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
            # self.line_raw, = self.axs[0].plot([], [])
            # self.scatter_pca = self.axs[1].scatter([], [], s=5, c='blue')
            #
            # self.axs[0].set_title("Raw Feature (Dims 0-2)")
            # self.axs[1].set_title("PCA Trajectory (Limit Cycle)")
            # self.vis_counter = 0
            ###########################################
            # Visualization Model Action
            ##########################################
            # self.mu_hist_len = 300
            # self.mu_buffer = deque(maxlen=self.mu_hist_len)
            # self.mu_pca = PCA(n_components=2)
            # self.fig_mu, self.ax_mu = plt.subplots(figsize=(6, 6))
            # self.ax_mu.set_title("Action (Mu) Phase Space")
            # plt.ion()
            # self.mu_vis_timer = 0
            self.actor_mlp

        def load(self, params):
            super().load(params)
            self._task_units = params['task_mlp']['units']
                
            self._task_activation = params['task_mlp']['activation']
            self._task_initializer = params['task_mlp']['initializer']
            return

        def initialize_record_file(self, file_path):
            CSV_HEADER = ['semantic_label', 'state_idx', 'frequency', 'phase_value']
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                with open(file_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
                print(f"Initialized new record file: {file_path}")
            else:
                print(f"Record file already exists: {file_path}")
        def append_record(self, file_path, text, indexes, f, p):
            try:
                semantic_label = str(text) if text is not None else "Unknown"

                state_idx = indexes.cpu().item() if indexes.ndim > 0 else indexes.cpu().item()

                frequency = f.cpu().item()
                phase_value = p.cpu().item()

                row = [semantic_label, state_idx, frequency, phase_value]
                with open(file_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

            except Exception as e:
                print(f"Error writing record: {e}")

        def get_phase_manifold(self, state, angles):
            """
            :param state: (batch_size, n_channel_latent)
            :param angles: (batch_size, n_channel_phase, time_range)
            :return:
            """
            state = state.reshape((state.shape[0], angles.shape[1], -1, 2))
            y0 = torch.cos(angles)
            y1 = torch.sin(angles)
            y = torch.stack((y0, y1), dim=-2)
            signal = y
            y = state @ y
            y = y.reshape(y.shape[0], -1, y.shape[-1])
            return y, signal

        def fft_with_nn(self, func, dim):
            rfft = torch.fft.rfft(func, dim=dim)
            magnitudes = rfft.abs()
            spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
            power = spectrum ** 2

            # Frequency
            freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

            # Amplitude
            amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

            # Offset
            offset = rfft.real[:, :, 0] / self.time_range  # DC component

            return freq, amp, offset

        def analytical_phase(self, latent, f, b, prior_mode=False):
            b = b.unsqueeze(-1)
            f = f.unsqueeze(-1)
            args = self.prior_args[1:-1] if prior_mode else self.args

            y = latent - b
            phase_term = self.tpi * f * args
            sx = torch.sum(y * torch.cos(phase_term), dim=2)
            sy = torch.sum(y * torch.sin(phase_term), dim=2)
            if torch.any((f.squeeze(-1) == 0) & (sx == 0)):
                print("Warning: f == 0 and sx == 0 detected. This may cause atan2(0, 0) and produce NaN gradients.")
            p = -torch.atan2(sy, sx + 1e-8) / self.tpi
            return p

        def pae(self, latent):
            latent1d = self.phase_conv(latent)
            f, a, b = self.fft_with_nn(latent1d, dim=2)
            p = self.analytical_phase(latent1d, f, b)
            return f, a, b, p

        def form_embedding(self, task_out_z, obs_dict = None):
            ###########################################
            # Visualization Normalized Observation
            ##########################################
            # try:
            #     if task_out_z.dim() == 3:
            #         last_frame = task_out_z[0, -1, :].detach().cpu().numpy()
            #     elif task_out_z.dim() == 2:
            #         last_frame = task_out_z[0, :].detach().cpu().numpy()
            #     else:
            #         last_frame = None
            #     if last_frame is not None:
            #         self.raw_data_buffer.append(last_frame)
            #         self.vis_counter += 1
            #         # Update plot every 5 steps to prevent lag
            #         if len(self.raw_data_buffer) > 66:
            #             data_arr = np.array(self.raw_data_buffer)  # Shape: [Time, Features]
            #             # Plot 1: Raw Data (First 3 dims to see rhythm)
            #             self.axs[0].clear()
            #             self.axs[0].plot(data_arr[:, :3])
            #             self.axs[0].set_title("Raw Features (Time)")
            #             # Plot 2: PCA Projection (The Manifold)
            #             if len(self.raw_data_buffer) > 20:
            #                 pca_res = self.pca_fitter.fit_transform(data_arr)
            #                 self.axs[1].clear()
            #                 # Color points by time (fading tail)
            #                 self.axs[1].plot(pca_res[:, 0], pca_res[:, 1], color='gray', linewidth=1, alpha=0.5)
            #                 colors = np.linspace(0, 1, len(pca_res))
            #                 self.axs[1].scatter(pca_res[:, 0], pca_res[:, 1], c=colors, cmap='viridis', s=10)
            #                 self.axs[1].set_title("PCA Trajectory (Phase Space)")
            #             plt.pause(0.001)
            #             import ipdb; ipdb.set_trace()
            # except Exception as e:
            #     print(f"Vis Error: {e}")

            extra_dict = {}
            if task_out_z.dim() == 2:
                B, N = task_out_z.shape
            elif task_out_z.dim() == 3:
                B, W, F = task_out_z.shape
            if self.z_type == 'vae':
                self.vae_mu = vae_mu = self.z_mu(task_out_z)
                self.vae_log_var = vae_log_var = self.z_logvar(task_out_z)
                
                if self.use_vae_clamped_prior:
                    self.vae_log_var = vae_log_var = torch.clamp(vae_log_var, min = -5, max = self.vae_var_clamp_max)
                
                if "z_noise"  in obs_dict and self.training: # bypass reparatzation and use the noise sampled during training. 
                    task_out_proj = vae_mu + torch.exp(0.5*vae_log_var) * obs_dict['z_noise']
                else:
                    task_out_proj, self.z_noise = self.reparameterize(vae_mu, vae_log_var)
                    
                if flags.test:
                    task_out_proj = vae_mu
                    
                if flags.trigger_input:
                    flags.trigger_input = False
                    flags.debug = not flags.debug
                    
                if flags.debug:
                    if self.use_vae_prior or self.use_vae_fixed_prior:
                        prior_mu, prior_logvar = self.compute_prior(obs_dict)
                        # if flags.trigger_input:
                        #     ### Trigger input
                        #     task_out_proj[:], noise = self.reparameterize(prior_mu, prior_logvar) ; print("\n   debugging",  end='')
                        #     flags.trigger_input = False
                        # else:
                        #     task_out_proj[:] = prior_mu
                        # task_out_proj[:], noise = self.reparameterize(prior_mu, torch.ones_like(prior_logvar) * -2.3 ) ; print("\r  debugging with prior using -2.3 std.",  end='')
                        # task_out_proj[:], noise = self.reparameterize(prior_mu, torch.ones_like(prior_logvar) * -1.5 ) ; print("\r  debugging with prior using -1.5 std.",  end='')
                        task_out_proj[:], noise = self.reparameterize(prior_mu, prior_logvar ) ; print(f"\r prior_mu {prior_mu.abs().max():.3f} {prior_logvar.exp().max():.3f}",  end='')
                        # task_out_proj[:] = torch.randn_like(vae_mu) ; print("\r   debugging randn",  end='')
                        enhance = 0
                    else:
                        task_out_proj[:] = torch.randn_like(vae_mu) ; print("\r   debugging",  end='')
                        
                if self.use_vae_sphere_posterior:
                    task_out_proj = project_to_norm(task_out_proj, norm=self.embedding_norm, z_type="sphere")
                
                extra_dict = {"vae_mu": vae_mu, "vae_log_var": vae_log_var, "noise": self.z_noise}
                
                
                # prior_mu, prior_logvar = self.compute_prior(obs_dict)
                # print(prior_logvar.exp().max())
                # np.set_printoptions(precision=4, suppress=1)
                # if "prev_task_out_proj" in self.__dict__:
                #     diff = self.prev_task_out_proj.cpu().numpy() - task_out_proj.cpu().numpy()
                #     print(f"{np.abs(diff).max():.4f}", diff)
                #     if np.abs(diff).max() > 0.5:
                #         import ipdb; ipdb.set_trace()
                #         print('...')
                # self.prev_task_out_proj = task_out_proj
                
                # prior_mu, prior_logvar = self.compute_prior(obs_dict)
                # import ipdb; ipdb.set_trace()
                # print(prior_mu.abs().argmax(), prior_mu.abs().max(), task_out_proj.abs().argmax(), task_out_proj.abs().max())
                
                # print(task_out_proj.abs().max(), task_out_proj.abs().argmax(), task_out_proj.cpu().numpy())
                # torch.exp(prior_logvar * 0.5), torch.exp(0.5 * vae_log_var)
                # print(torch.exp(prior_logvar * 0.5).mean(), torch.exp(0.5 * vae_log_var).mean())
                # print(task_out_proj.abs().max(), (prior_mu - task_out_proj).abs().cpu().numpy().max(), (prior_mu - task_out_proj).cpu().numpy())
                # import ipdb; ipdb.set_trace()
                
            elif self.z_type == 'vq_vae':
                z_before_quant = task_out_z
                # loss, task_out_proj, indexes = self.quantizer(project_to_norm(z_before_quant, norm=self.embedding_norm, z_type="sphere"))
                
                loss, task_out_proj, indexes = self.quantizer(z_before_quant.view(B, -1, self.embedding_size//self.embedding_partion))
                task_out_proj = task_out_proj.view(B, self.embedding_size)
                
                if flags.trigger_input:
                    flags.trigger_input = False
                    flags.debug = not flags.debug
                    # enhance = 0.5
                
                if flags.debug:
                    if flags.trigger_input:
                        indexes_input = input("Enter word indexes:")
                        try:
                            self.debug_idxes = [int(i) for i in indexes_input.split()]
                        except:
                            import ipdb; ipdb.set_trace()
                            pass 
                        flags.trigger_input = False
                    # import ipdb; ipdb.set_trace()
                    # self.debug_idxes =  self.embedding_size//self.embedding_partion, self.embedding_partion
                    prior_mu = self.compute_vq_prior(obs_dict)
                    _, z_out, _ = self.quantizer(prior_mu)
                    # indexes = torch.tensor(self.debug_idxes)
                    # embedding = self.quantizer.embedding.weight.data
                    # fixed_task_out_proj = torch.cat([embedding[self.debug_idxes[idx]] for idx in range(len(self.debug_idxes))])[None, ]; print("   debugging",  end='')
                    
                    if self.z_all: ## pass thorugh
                        fixed_task_out_proj = torch.cat([fixed_task_out_proj[:, :int(self.embedding_size * 3/4 )], task_out_proj[:, int(self.embedding_size * 3/4):]], dim=-1)    
                        
                    task_out_proj = z_out
                    
                if flags.test:
                    # print(f'\r {indexes[:self.embedding_partion].numpy()[12:16]}  ')
                    # print(f'\r { "".join([str(i) for i in indexes[:int(self.embedding_partion * 3/4)].numpy()]) } { "".join([str(i) for i in indexes[int(self.embedding_partion * 3/4):].numpy()]) }  ')
                    pass
                    # print(f'\r {indexes[:self.embedding_partion].numpy()}  {self.quantizer.embedding.weight.norm(dim = -1 ).data.numpy()}  ')
                    # print(f'\r {indexes[:self.embedding_partion].numpy()} {indexes.unique().numpy()} {self.quantizer.embedding.weight.norm(dim = -1 ).data.numpy()}  ')
                 
                else:
                    if flags.trigger_input:
                        import ipdb; ipdb.set_trace()
                        flags.trigger_input = False
                        print('...')
                
                extra_dict = {"loss": loss, "indexes": indexes, "z_before_quant": z_before_quant, "quantized_z_out": task_out_proj}
            elif self.z_type == 'vq_vae_hybrid':
                z_before_quant = self.z_quant(task_out_z)
                z_var = self.z_var(task_out_z)
                loss, task_out_proj, indexes= self.quantizer(z_before_quant)
                z_var = project_to_norm(z_var, norm=0.1, z_type="uniform")
                # loss += torch.norm(z_var, dim = -1).mean() 

                # task_out_proj = self.quantizer.embedding.weight.data[flags.idx % self.dict_size][None, ]; z_var[:] = 0; print("  debugging",  end='')
                # print(z_var)
                # print(f'\r {indexes[:3].numpy()} {indexes.unique().numpy()} {self.quantizer.embedding.weight.norm(dim = -1 ).data.numpy()}  ', end='')
                
                task_out_proj = torch.cat([task_out_proj, z_var], dim=-1)
                extra_dict = {"loss": loss, "indexes": indexes, "z_before_quant": z_before_quant, "quantized_z_out": task_out_proj}
            
            elif self.z_type == 'vq_vae_res':
                
                z_before_quant = self.z_quant(task_out_z)
                z_var = self.z_var(task_out_z)
                
                loss, task_out_proj, indexes = self.quantizer(project_to_norm(z_before_quant, norm=self.embedding_norm, z_type="sphere"))
                task_out_proj = project_to_norm(task_out_proj, norm= self.embedding_norm, z_type = "sphere")
                z_var  = torch.sin(z_var) + 1 # bias the number towards 1
                # loss += torch.norm(z_var , dim = -1).mean() 
                # task_out_proj = self.quantizer.embedding.weight.data[flags.idx % self.dict_size][None, ]; z_var[:] = 1; print("   debugging",  end='')
                
                task_out_proj = task_out_proj * z_var 
                print(f'\r {indexes[:3].numpy()} {indexes.unique().numpy()} {self.quantizer.embedding.weight.norm(dim = -1 ).data.numpy()}  ', end='')
                
                extra_dict = {"loss": loss, "indexes": indexes, "z_before_quant": z_before_quant, "quantized_z_out": task_out_proj}
            elif self.z_type == "sphere":
                task_out_proj = project_to_norm(task_out_z, norm=self.embedding_norm, z_type=self.z_type)
            elif self.z_type == "vq_pae":
                kinematic_obs_window = obs_dict['kinematic_obs_window'].transpose(1, 2)
                text_feat = self.text_adapter(obs_dict['clip_embedding'])
                latent = self.z_encoder(kinematic_obs_window)
                # ---- Phase Prediction ----
                f, a, b, p = self.pae(latent)

                ###############################################
                # Check Text label
                ###############################################
                # clip_embedding_cur = obs_dict['clip_embedding']
                #
                # def find_exact_embedding(target_emb, master_embeddings, master_texts):
                #
                #     target_emb = target_emb.view(1, -1)  # Make it (1, 512)
                #     equality_mask = torch.eq(target_emb, master_embeddings)
                #     match_index = torch.all(equality_mask, dim=1)
                #     indices = torch.nonzero(match_index, as_tuple=True)[0]
                #
                #     if indices.numel() > 0:
                #         first_match_index = indices[0].item()
                #         return master_texts[first_match_index]
                #     else:
                #         return None
                # text = find_exact_embedding(clip_embedding_cur, self.master_embeddings, self.master_texts)
                # print("Predicted text:", text)
                ###############################################
                state_input = latent.mean(axis=-1)
                state = self.state_fc(state_input)
                state_ori = state


                loss, state, indexes, perplexity = self.quantizer(state)
                # print(indexes, f, p)
                if flags.trigger_input:
                    flags.trigger_input = False
                    flags.debug = not flags.debug
                    # self.debug_index += 1
                # if not flags.debug:
                #     print(indexes, f, p)
                #     print(text, indexes, f, p)
                #     self.append_record(self.RECORD_FILE, text, indexes, f, p)
                if flags.debug:
                    if flags.reset:
                        self.debug_phase_p = torch.full((1, 1), self.bottom_phase, device='cuda')
                        flags.reset = not flags.reset
                        print("Reset Start Phase")
                    if flags.freq_inc:
                        self.debug_freq += 0.05
                        flags.freq_inc = not flags.freq_inc
                        print("current debug frequency", self.debug_freq)
                    elif flags.freq_dec:
                        self.debug_freq -= 0.05
                        flags.freq_dec = not flags.freq_dec
                        print("current debug frequency", self.debug_freq)

                    if flags.Index_dec:
                        self.debug_index -= 1
                        flags.Index_dec = not flags.Index_dec
                        print("current debug index", self.debug_index)
                    elif flags.Index_inc:
                        self.debug_index += 1
                        flags.Index_inc = not flags.Index_inc
                        print("current debug index", self.debug_index)

                    if flags.P_upper_inc:
                        self.top_phase += 0.05
                        flags.P_upper_inc = not flags.P_upper_inc
                        print("current phase range:", self.bottom_phase, self.top_phase)
                    elif flags.P_upper_dec:
                        self.top_phase -= 0.05
                        flags.P_upper_dec = not flags.P_upper_dec
                        print("current phase range:", self.bottom_phase, self.top_phase)
                    if flags.P_lower_inc:
                        self.bottom_phase += 0.05
                        flags.P_lower_inc = not flags.P_lower_inc
                        print("current phase range:", self.bottom_phase, self.top_phase)
                    elif flags.P_lower_dec:
                        self.bottom_phase -= 0.05
                        flags.P_lower_dec = not flags.P_lower_dec
                        print("current phase range:", self.bottom_phase, self.top_phase)
                    B = state.shape[0]
                      # 默认为 0
                    # print(self.debug_index, f, p)
                    # 为 batch 中的每个样本强制使用这个索引
                    debug_indices = torch.full((B,), fill_value=self.debug_index,
                                               dtype=torch.long, device=state.device)
                    state = self.quantizer.embedding(debug_indices)

                    f = torch.tensor([[self.debug_freq]], device=state.device)  # fixed frequency

                    if flags.text_change:
                        import ipdb;
                        ipdb.set_trace()
                        flags.text_change = False
                    clip_embedding = self.get_clip_embedding(self.input_text, self.clip_embedding_dict).unsqueeze(0).cuda()
                    text_feat_debug = self.text_adapter(clip_embedding)
                    prior_mu, prior_info = self.compute_vqpae_prior(obs_dict, f)
                    extra_dict = {'adapted_clip_embedding': text_feat_debug}
                    return prior_mu, extra_dict

                    self.debug_phase_p += f * 0.033  # increment per step (adjust step size)
                    self.debug_phase_p = torch.where(
                        self.debug_phase_p > self.top_phase,
                        torch.full_like(self.debug_phase_p, self.bottom_phase),  # 大于 0.15 时置为 -0.5
                        self.debug_phase_p  # 否则保持原值
                    )
                    p = self.debug_phase_p.clone()  # current phase offset
                angles = self.tpi * (f.unsqueeze(-1) * self.args + p.unsqueeze(-1))

                y, signal = self.get_phase_manifold(state, angles)
                manifold = y
                manifold_ori, _ = self.get_phase_manifold(state_ori, angles)
                recon_kin_obs_window = self.deconvs(y)

                central_frame = manifold.shape[-1] // 2
                extra_dict = {"loss": loss, "indexes": indexes, "z_before_quant": manifold_ori[..., central_frame],
                              "quantized_z_out": manifold[..., central_frame], "state_before_quant": state_ori,
                              "state_after_quant": state, "frequency": f,
                              "perplexity": perplexity,'recon_kin_obs_window': recon_kin_obs_window,
                              'adapted_clip_embedding': text_feat,
                              }
                return manifold, extra_dict

            # print(task_out_proj.max(), task_out_proj.min())
            return task_out_proj, extra_dict

        def get_clip_embedding(self, key: str, clip_embedding_dict) -> torch.Tensor:
            embedding = clip_embedding_dict.get(key)
            if embedding is None:
                print("key not found")
                import ipdb;
                ipdb.set_trace()
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).float()
            elif isinstance(embedding, (list, tuple)):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            return embedding

        def compute_prior(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            
            prior_latent = self.z_prior(self_obs)
            prior_mu = self.z_prior_mu(prior_latent)
            if self.use_vae_prior:
                prior_logvar = self.z_prior_logvar(prior_latent)
                if self.use_vae_clamped_prior:
                    prior_logvar = torch.clamp(prior_logvar, min = -5, max = self.vae_var_clamp_max)
                return prior_mu, prior_logvar
            elif self.use_vae_fixed_prior:
                if self.use_vae_sphere_prior:
                    return project_to_norm(prior_mu, z_type="sphere", norm = self.embedding_norm), torch.ones_like(prior_mu) * self.vae_prior_fixed_logvar
                else:
                    return prior_mu, torch.ones_like(prior_mu ) * self.vae_prior_fixed_logvar

        def compute_vq_prior(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]

            prior_latent = self.z_prior(self_obs)
            prior_mu = self.z_prior_mu(prior_latent)
            return prior_mu

        def compute_vqpae_prior(self, obs_dict, frequency):

            # self_obs = obs_dict['obs'][:, :self.kinematic_obs_size ]
            past_len = obs_dict['kinematic_obs_window'].shape[1] // 2
            self_kinematic_obs_window = obs_dict['kinematic_obs_window'][:, past_len-self.prior_time_range+2:past_len].clone()
            # use current obs to replace the kinematic obs
            # self_kinematic_obs_window[:,-1] = self_obs
            self_kinematic_obs_window = self_kinematic_obs_window.permute(0, 2, 1)

            prior_latent = self.prior_z_encoder(self_kinematic_obs_window)
            state_input = prior_latent.mean(axis=-1)
            state = self.prior_state_fc(state_input)
            # state_ori = state

            loss, state, indexes, _ = self.quantizer(state, freeze_codebook=True)
            prior_latent1d = self.prior_phase_conv(prior_latent)  # B, 1, W
            offset = torch.mean(prior_latent1d, dim=2)
            p = self.analytical_phase(prior_latent1d, frequency, offset, prior_mode=True)

            angles = self.tpi * (frequency.unsqueeze(-1) * self.prior_args + p.unsqueeze(-1))

            prior_manifold, _ = self.get_phase_manifold(state, angles)
            return prior_manifold, {
                "state": state,
                "vq_loss": loss,
            }

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, eps

        def eval_z(self, obs_dict):
            obs = obs_dict['obs']

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            z_out = self.z_mlp(obs)
            if self.proj_norm:
                z_out, extra_dict = self.form_embedding(z_out, obs_dict)
            return z_out
        
        def read_z(self, z):
            z_readout = self.z_reader_mlp(z)
            return z_readout
        
        def eval_critic(self, obs_dict):

            obs = obs_dict['obs']
            if obs.dim() == 3:
                obs = obs[:, -1, :]
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)

            self_obs = obs[:, :self.self_obs_size]
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here
            
            if self.has_rnn:
                c_out_in = c_out
                c_out = self.critic_z_mlp(c_out_in)

                if self.rnn_concat_input:
                    c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = c_out.size()[0]
                num_seqs = batch_size // seq_length
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                ################# New RNN
                if len(states) == 2:
                    c_states = states[1].reshape(num_seqs, seq_length, -1)
                else:
                    c_states = states[2:].reshape(num_seqs, seq_length, -1)
                c_out, c_states = self.c_rnn(c_out, c_states[:, 0:1].transpose(0, 1).contiguous()) # ZL: only pass the first state, others are ignored. ???            
                
                ################# Old RNN
                # if len(states) == 2:	
                #     c_states = states[1]	
                # else:	
                #     c_states = states[2:]	
                # c_out, c_states = self.c_rnn(c_out, c_states)
                
                
                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if type(c_states) is not tuple:
                    c_states = (c_states,)
                
                c_out = self.critic_z_proj_linear(c_out)
                # c_out, extra_dict = self.form_embedding(c_out) # do not form VAE embedding for cirtic. 
                if self.z_type == "sphere":
                    c_out = project_to_norm(c_out, norm=self.embedding_norm, z_type=self.z_type)
                
                c_out = torch.cat([self_obs, c_out], dim=-1)

                c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                task_out = self.critic_z_mlp(obs)
                
                # c_out, extra_dict = self.form_embedding(c_out) # do not form VAE embedding for cirtic. 
                if self.z_type == "sphere": # but we do project for z sphere....
                    task_out = project_to_norm(task_out, norm=self.embedding_norm, z_type=self.z_type)
                    
                if self.z_all:
                    c_input = task_out
                else:
                    c_input = torch.cat([self_obs, task_out], dim=-1)
                c_out = self.critic_mlp(c_input)
                value = self.value_act(self.value(c_out))
                return value
            
        def eval_actor(self, obs_dict, return_extra = False):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)

            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            self_obs = obs[:, ..., :self.self_obs_size]
            task_root_obs = self.extract_root_task_condition(obs_dict['obs_orig'][:, self.self_obs_size:])
            assert (obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            
            if self.has_rnn:
                
                a_out_in = a_out
                
                a_out = self.z_mlp(obs)
                    
                if self.rnn_concat_input:
                    a_out = torch.cat([a_out, a_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)

                ################# New RNN
                if len(states) == 2:
                    a_states = states[0].reshape(num_seqs, seq_length, -1)
                else:
                    a_states = states[:2].reshape(num_seqs, seq_length, -1)
                a_out, a_states = self.a_rnn(a_out, a_states[:, 0:1].transpose(0, 1).contiguous())
                
                ################ Old RNN
                # if len(states) == 2:	
                #     a_states = states[0]	
                # else:	
                #     a_states = states[:2]	
                # a_out, a_states = self.a_rnn(a_out, a_states)
                

                if self.rnn_name == 'sru':
                    a_out = a_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)

                a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                
                z_out = self.z_proj_linear(a_out)
                
                if self.proj_norm:
                    z_out, extra_dict = self.form_embedding(z_out, obs_dict)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    
                actor_input = torch.cat([self_obs, z_out], dim=-1)
                a_out = self.actor_mlp(actor_input)

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, a_states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, a_states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                    
                    if return_extra:
                        return mu, sigma, a_states, extra_dict
                    else:
                        return mu, sigma, a_states
            else:
                # if self.z_all:
                #     task_out_z = self.z_mlp(task_obs)
                #     self_out_z = self.z_self_mlp(self_obs)
                #     # self_out_z[:] = 0
                #     task_out_z = torch.cat([task_out_z, self_out_z], dim=-1)
                # else:
                #     task_out_z = self.z_mlp(obs)

                if self.z_type != 'vq_pae':
                    task_out_z = self.z_mlp(obs)
                else:
                    task_out_z = obs

                if self.proj_norm:
                    z_out, extra_dict = self.form_embedding(task_out_z, obs_dict)
                
                # if "z_acc" not in self.__dict__.keys():
                #     self.z_acc = []
                # self.z_acc.append(z_out)
                # if len(self.z_acc) > 500:
                #     import ipdb; ipdb.set_trace()
                #     import joblib;joblib.dump(self.z_acc, "z_acc_compare_3.pkl")
                if self.z_all:
                    actor_input = z_out
                else:
                    central_frame = -1 if z_out.shape[-1] == self.prior_time_range else z_out.shape[-1] // 2   # 61 // 2 = 30
                    # print(task_root_obs)
                    # import math
                    # angle = math.radians(0)
                    # task_root_obs[:, 0] = -0.100
                    # task_root_obs[:, 1] = 0.00
                    # task_root_obs[:, 2] = math.cos(angle)
                    # task_root_obs[:, 3] = math.sin(angle)
                    actor_input = torch.cat([self_obs, task_root_obs, z_out[:, :, central_frame], extra_dict['adapted_clip_embedding']], dim=-1) # [B, Window, Feature]

                a_out = self.actor_mlp(actor_input)
                
                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits
                
                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    ###########################################
                    # Visualization Model Action
                    ##########################################
                    # self._visualize_mu(mu)
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))
                        
                    if return_extra:
                        return mu, sigma, extra_dict
                    else:
                        return mu, sigma

        def extract_root_task_condition(self, task_obs, num_joints=24, time_steps=1):
            """
            Extracts the specific Root (Joint 0) information from the concatenated task_obs.

            Args:
                task_obs: The concatenated observation tensor.
                          Shape: (B, time_steps * total_features) or (B, total_features) if flattened.
                num_joints: Number of joints (default 24 based on your code).
                time_steps: The length of the window (T).

            Returns:
                root_condition: A tensor containing only the root's target info and velocity differences.
                                Shape: (B, -1) which flattens (B, time_steps, 15)
            """
            B = task_obs.shape[0]
            obs_reshaped = task_obs.view(B, time_steps, -1)

            dim_pos = 3  # diff_local_body_pos (x,y,z)
            dim_rot = 6  # diff_local_body_rot (6d rotation)
            dim_vel = 3  # diff_local_vel
            dim_ang = 3  # diff_local_ang_vel
            dim_ref_pos = 3  # local_ref_body_pos
            dim_ref_rot = 6  # local_ref_body_rot

            idx_diff_pos = 0

            idx_diff_rot = idx_diff_pos + (num_joints * dim_pos)
            idx_diff_vel = idx_diff_rot + (num_joints * dim_rot)
            idx_diff_ang = idx_diff_vel + (num_joints * dim_vel)
            idx_ref_pos = idx_diff_ang + (num_joints * dim_ang)
            idx_ref_rot = idx_ref_pos + (num_joints * dim_ref_pos)

            root_diff_pos = obs_reshaped[:, :, 0: 3]
            root_diff_rot = obs_reshaped[:, :, idx_diff_rot: idx_diff_rot + 6]
            root_diff_vel = obs_reshaped[:, :, idx_diff_vel: idx_diff_vel + 3]
            root_diff_ang = obs_reshaped[:, :, idx_diff_ang: idx_diff_ang + 3]
            root_ref_pos = obs_reshaped[:, :, idx_ref_pos: idx_ref_pos + 2]
            root_ref_rot = obs_reshaped[:, :, idx_ref_rot: idx_ref_rot + 6]

            ref_tan_in_local = root_ref_rot[:, 0, :3]
            ref_heading_2d = ref_tan_in_local[..., :2]
            ref_heading_2d_rot = torch.nn.functional.normalize(ref_heading_2d, dim=-1).view(B, time_steps, -1)
            root_task_condition = torch.cat([
                # root_diff_pos,
                # root_diff_rot,
                # root_diff_vel,  # Velocity Correction
                # root_diff_ang,  # Turning Correction
                root_ref_pos,  # Target Displacement (Most Important)
                # root_ref_rot,  # Target Orientation
                ref_heading_2d_rot
            ], dim=-1)

            return root_task_condition.view(B, -1)

        def _visualize_mu(self, mu):
            """
            Projects the high-dim action mean (mu) to 2D PCA to check for periodicity.
            """
            try:
                # Handle shapes: [Batch, Dim] or [Batch, Time, Dim]
                # We always take Batch 0, Last Time Step
                if mu.dim() == 2:
                    current_mu = mu[0].detach().cpu().numpy()
                elif mu.dim() == 3:
                    current_mu = mu[0, -1, :].detach().cpu().numpy()
                else:
                    return

                self.mu_buffer.append(current_mu)
                self.mu_vis_timer += 1

                # Update plot every 5 steps
                if len(self.mu_buffer) > 65:
                    data = np.array(self.mu_buffer)

                    # Fit PCA
                    projected = self.mu_pca.fit_transform(data)

                    self.ax_mu.clear()
                    self.ax_mu.plot(projected[:, 0], projected[:, 1], color='gray', alpha=0.3, linewidth=1)

                    # 2. Draw the dots (Time)
                    colors = np.linspace(0, 1, len(projected))
                    self.ax_mu.scatter(projected[:, 0], projected[:, 1], c=colors, cmap='plasma', s=20)

                    # 3. Draw Digital Numbers (Indices)
                    # We iterate through the points and add text labels
                    total_points = len(projected)
                    for i in range(total_points):
                        # Label the first point, the last point, and every 5th point in between
                        if i == 0 or i == total_points - 1 or i % 5 == 0:
                            self.ax_mu.text(
                                projected[i, 0],
                                projected[i, 1],
                                str(i),  # The number to draw
                                fontsize=9,
                                color='black',
                                fontweight='bold' if i == total_points - 1 else 'normal'
                            )

                    self.ax_mu.set_title(f"Action Mean Trajectory (Step {self.mu_vis_timer})")
                    plt.pause(0.001)
                    import ipdb; ipdb.set_trace()
            except Exception as e:
                print(f"Mu Vis Error: {e}")

        def _build_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            
            if self.z_type == "vae" or self.z_type == "vq_vae_hybrid" or self.z_type == "vq_vae_res":
                out_size = self.embedding_size * 5
            else:
                # if self.z_all:
                #     out_size = int(self.embedding_size  * 3/4 )
                # else:
                #     out_size = self.embedding_size
                out_size = self.embedding_size

            # if self.z_all:
            #     mlp_input_shape = task_obs_size
            # else:
            #     mlp_input_shape = self_obs_size + task_obs_size  # target
            
            mlp_input_shape = self_obs_size + task_obs_size  # target

            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.z_mlp = self._build_mlp(**mlp_args)
            
            if not self.has_rnn:
                self.z_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=out_size))
            else:
                self.z_proj_linear = nn.Linear(in_features=self.rnn_units, out_features=out_size)
            
            mlp_init = self.init_factory.create(**self._task_initializer)
            init_mlp(self.z_mlp, mlp_init)
            
            # if self.z_all:
            #     mlp_args = {'input_size': self_obs_size, 'units': self._self_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            #     self.z_self_mlp = self._build_mlp(**mlp_args)
            #     if not self.has_rnn:
            #         self.z_self_mlp.append(nn.Linear(in_features=self._self_units[-1], out_features=int(self.embedding_size  * 1/4 )))
            #     else:
            #         self.z_self_proj_linear = nn.Linear(in_features=self.rnn_units, out_features=int(self.embedding_size  * 1/4 ))

                        
            if self.z_type == "vae":
                self.z_mu = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                self.z_logvar = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                
                init_mlp(self.z_mu, mlp_init); init_mlp(self.z_logvar, mlp_init)
                
                if self.use_vae_prior:
                    mlp_args = {'input_size': self_obs_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                    self.z_prior = self._build_mlp(**mlp_args)
                    self.z_prior_mu = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                    self.z_prior_logvar = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                    init_mlp(self.z_prior, mlp_init); init_mlp(self.z_prior_mu, mlp_init); init_mlp(self.z_prior_logvar, mlp_init)
                    
                    # import ipdb; ipdb.set_trace()
                    # print('..... Disabling prior training ......')
                    # print('..... Disabling prior training ......')
                    # print('..... Disabling prior training ......')
                    # self.z_prior.requires_grad_(False)
                    # self.z_prior_mu.requires_grad_(False)
                    # self.z_prior_logvar.requires_grad_(False)

                elif self.use_vae_fixed_prior:
                    mlp_args = {'input_size': self_obs_size, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                    self.z_prior = self._build_mlp(**mlp_args)
                    self.z_prior_mu = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                    init_mlp(self.z_prior, mlp_init); init_mlp(self.z_prior_mu, mlp_init)
                    
            elif self.z_type == 'vq_vae':
                self.quantizer = Quantizer(self.dict_size, self.embedding_size//self.embedding_partion, 0.25)
                # self.quantizer = EMAVectorQuantizer(self.dict_size, self.embedding_size//4, 0.25, decay = 0.99)
                mlp_args = {'input_size': self_obs_size, 'units': self._task_units, 'activation': self._task_activation,
                            'dense_func': torch.nn.Linear}
                self.z_prior = self._build_mlp(**mlp_args)
                self.z_prior_mu = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                # self.z_prior_logvar = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
                init_mlp(self.z_prior, mlp_init)
                init_mlp(self.z_prior_mu, mlp_init)
                # init_mlp(self.z_prior_logvar, mlp_init)
            elif self.z_type == 'vq_pae':
                self.clip_dim = getattr(self, 'clip_dim', 512)
                self.n_input_channels = self.kinematic_obs_size
                self.n_latent_channels = self.embedding_size
                self.fps = 30.
                self.window = getattr(self, 'window', (self.window_size - 1) / self.fps) # window=1.0, 2.0
                self.time_range = self.window_size # time_range: 31, 61, 121 ..
                from torch.nn.parameter import Parameter
                self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window,
                                       requires_grad=False)  # Remove DC frequency
                self.n_timing_phases = getattr(self, 'n_timing_phases', 1)

                self.intermediate_channels = getattr(self, 'intermediate_channels', 128)
                self.pae_n_layers = getattr(self, 'pae_n_layers', 2)
                self.pae_kernel_size = getattr(self, 'pae_kernel_size', 5)
                self.pae_n_layers_fft = getattr(self, 'pae_n_layers_fft', 7)
                n_layers_state = getattr(self, 'pae_n_layers_state', 5)

                self.num_embed = 2 * self.n_latent_channels

                self.tpi = nn.Parameter(torch.tensor(2 * np.pi, dtype=torch.float32), requires_grad=False)
                self.args = nn.Parameter(
                    torch.from_numpy(np.linspace(-self.window / 2, self.window / 2, self.time_range,
                                                 dtype=np.float32)), requires_grad=False)
                self.prior_time_range = self.prior_window_size
                self.prior_window = (self.prior_time_range - 1) / self.fps
                self.prior_args = nn.Parameter(
                    torch.from_numpy(np.linspace(-self.prior_window / 2, self.prior_window / 2, self.prior_time_range,
                                                 dtype=np.float32)), requires_grad=False)

                encoder_channels = [self.n_input_channels] + [self.intermediate_channels] * (self.pae_n_layers - 1) + [self.n_latent_channels]
                normalizer = partial(LN_v3, keep_std=True)
                self.z_encoder = []
                for i in range(self.pae_n_layers):
                    self.z_encoder.append(nn.Conv1d(encoder_channels[i], encoder_channels[i + 1],
                                                    self.pae_kernel_size, padding='same'))
                    self.z_encoder.append(normalizer(self.time_range)) # Requires normalizer
                    self.z_encoder.append(nn.ELU())
                self.z_encoder = nn.Sequential(*self.z_encoder)

                # ---- 2. Phase Convolution ----
                # Takes [B, pae_latent_channels, W] -> [B, n_timing_phases, W]
                self.phase_conv = nn.Sequential(nn.Conv1d(self.n_latent_channels, self.n_timing_phases, self.pae_kernel_size, padding='same'))

                # ---- 3. Frequency MLP (from FFT) ----
                fft_in_length = self.window_size // 2 + 1
                self.freq_fc = MLP(self.pae_n_layers_fft, fft_in_length, 1, 1, bn=False, last_activation=True)

                # ---- 4. State MLP (from latent mean) ----
                # Input is latent.mean(dim=-1), shape [B, pae_latent_channels]
                # Output is shape [B, pae_state_dim] (which is self.embedding_size)
                 # define how many FC layers (configurable)
                self.text_adapter = nn.Sequential(
                    nn.Linear(self.clip_dim, self.clip_dim),  # [B, 7, 512] -> [B, 7, 512]
                )
                n_channels_state_mlp = [self.n_latent_channels] + [self.num_embed] * n_layers_state
                self.state_fc = MLPChannels(n_channels_state_mlp, bn=False)
                self.state_proj_head = nn.Linear(self.num_embed, 512)
                # ---- 5. Vector Quantizer ----
                self.quantizer = VectorQuantizer(self.dict_size, self.num_embed, 0.25)

                self.deconvs = []
                decoder_channels = encoder_channels[::-1]
                for i in range(self.pae_n_layers):
                    self.deconvs.append(nn.Conv1d(decoder_channels[i], decoder_channels[i + 1],
                                                  self.pae_kernel_size, padding='same'))
                    if i != self.pae_n_layers - 1:
                        self.deconvs.append(normalizer(self.window_size))  # Use window_size
                        self.deconvs.append(nn.ELU())
                self.deconvs = nn.Sequential(*self.deconvs)
                # init_mlp(self.deconvs, mlp_init)  # Initialize the decoder
                ###############################
                # prior
                ###############################
                self.prior_input_channels = self.kinematic_obs_size
                prior_encoder_channels = [self.prior_input_channels] + [self.intermediate_channels] * (
                            self.pae_n_layers - 1) + [self.n_latent_channels]
                self.prior_z_encoder = []
                for i in range(self.pae_n_layers):
                    self.prior_z_encoder.append(nn.Conv1d(prior_encoder_channels[i], prior_encoder_channels[i + 1],
                                                          self.pae_kernel_size, padding='same'))
                    self.prior_z_encoder.append(normalizer(self.time_range))  # Requires normalizer
                    self.prior_z_encoder.append(nn.ELU())
                self.prior_z_encoder = nn.Sequential(*self.prior_z_encoder)
                self.prior_phase_conv = nn.Sequential(
                    nn.Conv1d(self.n_latent_channels, self.n_timing_phases, self.pae_kernel_size, padding='same'))

                prior_state_input_dim = self.n_latent_channels
                n_channels_prior_state_mlp = [prior_state_input_dim] + [self.num_embed] * n_layers_state
                self.prior_state_fc = MLPChannels(n_channels_prior_state_mlp, bn=False)
            elif self.z_type == 'vq_vae_hybrid':
                self.z_quant = nn.Linear(in_features=self.embedding_size * 5, out_features=int(self.embedding_size - 1))
                self.z_var = nn.Linear(in_features=self.embedding_size * 5, out_features=int(1))
                

                # mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
                # self.z_var = self._build_mlp(**mlp_args)
                # self.z_var.append(nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size))
                
                init_mlp(self.z_quant, mlp_init); init_mlp(self.z_var, mlp_init)
                self.quantizer = Quantizer(self.dict_size, int(self.embedding_size - 1), 0.25)

            elif self.z_type == 'vq_vae_res':
                self.z_quant = nn.Linear(in_features=self.embedding_size * 5, out_features=self.embedding_size)
                self.z_var = nn.Linear(in_features=self.embedding_size * 5, out_features=1)
                
                self.quantizer = Quantizer(self.dict_size, self.embedding_size, 0.25)
                init_mlp(self.z_quant, mlp_init); init_mlp(self.z_var, mlp_init)
            return
        
        def _build_critic_z_mlp(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self_obs_size + task_obs_size  # target

            self.critic_z_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.critic_z_mlp = self._build_mlp(**mlp_args)
            
            if not self.has_rnn:
                self.critic_z_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size))
            else:
                self.critic_z_proj_linear = nn.Linear(in_features=self._task_units[-1], out_features=self.embedding_size)
            

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.critic_z_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return

        def _build_z_reader(self):
            self_obs_size, task_obs_size, task_obs_size_detail = self.self_obs_size, self.task_obs_size, self.task_obs_size_detail
            mlp_input_shape = self.embedding_size  # target

            self.z_reader_mlp = nn.Sequential()
            mlp_args = {'input_size': mlp_input_shape, 'units': self._task_units, 'activation': self._task_activation, 'dense_func': torch.nn.Linear}
            self.z_reader_mlp = self._build_mlp(**mlp_args)
            self.z_reader_mlp.append(nn.Linear(in_features=self._task_units[-1], out_features=72))

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self.z_reader_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return
