import logging
import os
import math

import numpy as np
import json

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len) #linspace(start, stop, number)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] #returns residual for a polynomial fit for x pos
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1] #same for y pos
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=10, pred_len=20, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir) #listing all files in a directory
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        agent_considered = 0
        start_end = []
        for path in all_files:
            with open(path) as f:
                d = json.load(f)
            agent_pasts = np.asarray(d['agent_pasts'])
            agent_futures = np.asarray(d['agent_futures'])
            player_past = np.asarray(d['player_past'])
            player_future = np.asarray(d['player_future'])
            total_seq = np.concatenate((agent_pasts[:,:,:2], agent_futures[:,:,:2]), axis=1)
            start = agent_considered
            num_agent = agent_pasts.shape[0]
            agent_considered += num_agent
            start_end.append((start,agent_considered))
            total_seq_re = []
            for i in range(total_seq.shape[0]):
                total_seq_re.append(total_seq[i,:,:].T)
            player_seq = np.concatenate((player_past[:,:2], player_future[:,:2]), axis=0)
            total_seq_re = np.asarray(total_seq_re)
            seq_list.append(total_seq_re)
        seq_list = np.vstack(seq_list)
        seq_list_rel = np.zeros(seq_list.shape)
        non_linear_ped = np.zeros(seq_list.shape[0])
        for i in range(seq_list.shape[0]):
            seq_list_rel[i,:,1:] = seq_list[i,:,1:] - seq_list[i,:,:-1]
            non_linear_ped[i] = poly_fit(seq_list[i],20,threshold)
        loss_mask_list = np.ones((seq_list.shape[0],seq_list.shape[2]))

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.seq_start_end = start_end
        self.num_seq = len(start_end)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
