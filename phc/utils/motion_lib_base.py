
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import os
import yaml
from tqdm import tqdm

from phc.utils import torch_utils
import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import gc
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from enum import Enum
USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)
import json, pickle

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


class DeviceCache:

    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out

class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibBase():

    def __init__(self, motion_lib_cfg):
        self.m_cfg = motion_lib_cfg
        self._device = self.m_cfg.device
        
        self.mesh_parsers = None

        self.babel_ann = {}
        self.babel_lookup = {}
        self.babel_id_lookup = {}
        self.clip_embedding_dict = {}
        base_dir = os.path.dirname(self.m_cfg.motion_file)
        babel_dir = os.path.join(base_dir, "babel_v1.0_release")
        clip_pkl_path = os.path.join(base_dir, "text_embedding_dict_clip.pkl")
        # load text to embedding pkl file
        if os.path.exists(clip_pkl_path):
            print(f"Loading CLIP embeddings from: {clip_pkl_path} ...")
            with open(clip_pkl_path, 'rb') as f:
                self.clip_embedding_dict = joblib.load(f)
        else:
            print(f"Warning: CLIP embedding file NOT found at {clip_pkl_path}")
            import ipdb; ipdb.set_trace()
        # load babel labels file
        files = ["train.json", "val.json"]
        if os.path.exists(babel_dir):
            print(f"Loading BABEL from {babel_dir}...")
            for fname in files:
                fpath = os.path.join(babel_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        self.babel_ann.update(json.load(f))

            print("Building BABEL lookup table...")
            for sid, data in self.babel_ann.items():
                feat_p = data.get('feat_p', '')
                if not feat_p:
                    import ipdb; ipdb.set_trace()

                path_no_ext = os.path.splitext(feat_p)[0]
                key_variant_a = path_no_ext.replace('/', '_')

                parts = path_no_ext.split('/')
                if len(parts) > 1 and parts[0] == parts[1]:
                    key_variant_b = "_".join(parts[1:])
                else:
                    key_variant_b = key_variant_a

                frame_ann = data.get('frame_ann', {})
                if frame_ann is None:
                    frame_ann = data.get('seq_ann', {})
                labels = frame_ann.get('labels', [])
                self.babel_lookup[key_variant_b] = labels
                self.babel_id_lookup[key_variant_b] = data.get('babel_sid', '')
            print(f"BABEL lookup table built with {len(self.babel_lookup)} keys.")
        else:
            print("Warning: BABEL directory not found!")
            import ipdb; ipdb.set_trace()
        self.load_data(self.m_cfg.motion_file,  min_length = self.m_cfg.min_length, im_eval = self.m_cfg.im_eval)
        self.setup_constants(fix_height = self.m_cfg.fix_height,  multi_thread = self.m_cfg.multi_thread)

        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return
        
    def load_data(self, motion_file,  min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))

        filtered_data = {}
        raw_data = self._motion_data_load if self.mode == MotionlibMode.file else {}

        if self.mode == MotionlibMode.file:
            print(f"Filtering data... (Total raw: {len(raw_data)})")
            for k, v in tqdm(raw_data.items()):
                if len(self.babel_lookup) > 0:
                    clean_key = k.split('-', 1)[1] if '-' in k else k

                    if clean_key in self.babel_lookup:
                        num_frames = len(v['pose_quat_global'])
                        fps = v['fps']
                        curr_len = len(v['pose_quat_global']) / fps
                        found_id = self.babel_id_lookup[clean_key]
                        duration = self.babel_ann[str(found_id)]['dur']
                        if abs(duration - curr_len) > 1e-2:
                            print("Warning: The duration is not match; The original motion might be clipped by convert_amass_data.py bound variable!")
                        curr_text_dense = ["transition"] * num_frames
                        found_segments = self.babel_lookup[clean_key]
                        if found_segments:
                            for seg in found_segments:
                                label = seg.get('proc_label') or seg.get('raw_label') or seg.get('act_cat')
                                if label is None:
                                    print("Couldn't find the label")
                                    import ipdb; ipdb.set_trace()

                                if 'start_t' in seg and 'end_t' in seg:
                                    start_f = int(seg['start_t'] * fps)
                                    end_f = int(seg['end_t'] * fps)

                                    start_f = max(0, start_f)
                                    end_f = min(num_frames, end_f)

                                    for i in range(start_f, end_f):
                                        curr_text_dense[i] = label
                                else:
                                    # print("process segment level label")
                                    curr_text_dense = [label] * num_frames
                                    # import ipdb; ipdb.set_trace()
                        # post process transition label
                        last_meaningful_action = None
                        # here we first the first valid action to avoid the last frame is "transition" and
                        # we want it to transition to the first valid action as your cyclic motion pattern
                        for i in range(num_frames):
                            if curr_text_dense[i] != "transition":
                                last_meaningful_action = curr_text_dense[i]
                                break

                        if last_meaningful_action is None:
                            import ipdb; ipdb.set_trace() # robust check

                        for i in range(num_frames - 1, -1, -1):
                            current_label = curr_text_dense[i]

                            if current_label == "transition":
                                new_label = f"transition to {last_meaningful_action}"
                                curr_text_dense[i] = new_label
                            else:
                                last_meaningful_action = current_label
                        v['babel_text_labels'] = curr_text_dense

                        frame_embeddings = []
                        no_embedding = False
                        no_embedding_value = None
                        for lbl in curr_text_dense:
                            # Use the helper function to get the tensor
                            if lbl == 'transition to move head in a circle':
                                import ipdb; ipdb.set_trace()
                            emb = self.get_clip_embedding(lbl)
                            if emb is None:
                                no_embedding = True
                                no_embedding_value = lbl
                                break
                            frame_embeddings.append(emb)

                        if no_embedding:
                            print(f"Error: Couldn't find the clip embedding of {no_embedding_value}")
                            continue
                        if len(frame_embeddings) > 0:
                            clip_tensor = torch.stack(frame_embeddings)
                            v['clip_embeddings'] = clip_tensor.cpu()
                        else:
                            print("Couldn't find the frame embedding")
                            import ipdb; ipdb.set_trace()
                    else:
                        # print("do not have matched babel data!", clean_key)
                        continue

                filtered_data[k] = v
            self._motion_data_load = filtered_data
            print(f"Motions remaining after BABEL filter: {len(self._motion_data_load)}")

            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches
        
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        raise NotImplementedError

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        raise NotImplementedError

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs,
            del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
            if hasattr(self, '_motion_clip_embeddings'):
                del self._motion_clip_embeddings
            if flags.real_traj:
                del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []
        self._motion_clip_embeddings = []

        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)
            # sample_idxes = torch.tensor([3], device=self._device)
            # sample_idxes = torch.randint(
            #     low=0,
            #     high=len(self._motion_data_keys),
            #     size=(1,),
            #     device=self._device
            # )

        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:30])
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")


        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 64)

        if num_jobs <= 8 or not self.multi_thread:
            num_jobs = 1
        if flags.debug:
            num_jobs = 1
        
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk],  self.mesh_parsers, self.m_cfg) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            
            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                self._motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                self._motion_bodies.append(torch.zeros(17))

            if 'clip_embeddings' in motion_file_data:
                self._motion_clip_embeddings.append(motion_file_data['clip_embeddings'])
            else:
                print("Could not find clip_embeddings!")
                import ipdb; ipdb.set_trace()

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(self._motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.clip_embeddings = torch.cat(self._motion_clip_embeddings, dim=0).to(self._device)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_clip_embedding(self, key: str):
        """
        Retrieves the corresponding CLIP embedding tensor for a given text key.

        If the key is found, the value is converted to a PyTorch Tensor and moved to the correct device.
        If the key is not found, a zero vector of the correct dimensionality is returned as a fallback.

        Args:
            key (str): The text label (e.g., 'walk', 'jump') used to look up the embedding.

        Returns:
            torch.Tensor: The embedding vector (e.g., shape [512]) on self._device.
        """
        # if key == 'transition to move head in a circle':
        #     key = 'transition to move head around'
        embedding = self.clip_embedding_dict.get(key)

        if embedding is None:
            return embedding

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        elif isinstance(embedding, (list, tuple)):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        return embedding
    # def update_sampling_weight(self):
    #     ## sampling weight based on success rate. 
    #     # sampling_temp = 0.2
    #     sampling_temp = 0.1
    #     curr_termination_prob = 0.5

    #     curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids] / self._sampling_history[self._curr_motion_ids]
    #     self._success_rate[self._curr_motion_ids] = curr_succ_rate
    #     sample_prob = torch.exp(-self._success_rate / sampling_temp)

    #     self._sampling_prob = sample_prob / sample_prob.sum()
    #     self._termination_history[self._curr_motion_ids] = 0
    #     self._sampling_history[self._curr_motion_ids] = 0

    #     topk_sampled = self._sampling_prob.topk(50)
    #     print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])
        
    def update_hard_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._sampling_prob[:] = 0
            self._sampling_prob[indexes] = 1/len(indexes)
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training on only {len(failed_keys)} seqs")
            print(failed_keys)
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
            
    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)    
            
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training mostly on {len(self._sampling_prob.nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.nonzero()].flatten())
            print(f"###############################################################################################################################")
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history/termination_history.sum()
            self._termination_history = termination_history
            return True
        else:
            return False
        
        
    # def update_sampling_history(self, env_ids):
    #     self._sampling_history[self._curr_motion_ids[env_ids]] += 1
    #     # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    # def update_termination_history(self, termination):
    #     self._termination_history[self._curr_motion_ids] += termination
    #     # print("termination history: ", self._termination_history[self._curr_motion_ids])

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * 30 / self._motion_fps).int()
        else:
            return (self._motion_num_frames[motion_ids] * 30 / self._motion_fps).int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        clip_emb = self.clip_embeddings[f0l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1


        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
            "clip_embedding": clip_emb.clone()  # 使用 clone() 以防外部修改影响原始数据
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)