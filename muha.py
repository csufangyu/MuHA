


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from einops import rearrange
from copy import deepcopy
from rl_utils import compute_attribution,compute_attribution_mask,\
                    make_obs_grid,compute_guided_actor,compute_guided_full_model
import random
import kornia as K
from kornia.augmentation import ImageSequential
from captum.attr import GuidedBackprop
def random_mask_freq_v2(x,keep=0.5):
    p = random.uniform(0, 1)
    if p > keep:
        return x
    A = 0
    B = 0.5
    a = random.uniform(A, B)
    C = 2
    freq_limit_low = round(a, C)
    A = 0
    B = 0.05
    a = random.uniform(A, B)
    C = 2
    diff = round(a, C)
    freq_limit_hi = freq_limit_low + diff

    # b, 9, h, w
    b, c, h, w = x.shape
    x0, x1, x2 = torch.chunk(x, 3, dim=1)
    # b, 3, 3, h, w
    x = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_hi
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_hi
    kernel1 = torch.outer(pass2, pass1)  # freq_limit_hi square is true

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_low
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_low
    kernel2 = torch.outer(pass2, pass1)  # freq_limit_low square is true

    kernel = kernel1 * (~kernel2)  # a square ring is true
    fft_1 = torch.fft.fftn(x, dim=(2, 3, 4))
    imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2, 3, 4)).float()
    x0, x1, x2 = torch.chunk(imgs, 3, dim=1)
    imgs = torch.cat((x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)), dim=1)
    return imgs




class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val = 'inside_random', spacing='random', transforms_like=False, p_random_apply_rand_quant = 1):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
        return x



def random_aug(x,random_quant=None):
    N = x.size(0)   
    enhanced_tensor = torch.empty_like(x).to(x.device)
    indices = torch.randperm(N).to(x.device)
    quarter = N // 5
    group1_idx = indices[:quarter]
    group2_idx = indices[quarter: 2 * quarter]
    group3_idx = indices[2 * quarter: 3 * quarter]
    group4_idx = indices[3 * quarter: 4 * quarter]
    group5_idx = indices[4 * quarter:]
    group1 = x[group1_idx]
    group2 = x[group2_idx]
    group3 = x[group3_idx]
    group4 = x[group4_idx]
    group5 = x[group5_idx]
    group1 = random_quant(group1)
    group2 = random_quant(random_mask_freq_v2(group2))
    group3 = random_mask_freq_v2(group3)
    group4 = random_quant(group4)
    group5 = random_mask_freq_v2(group5)
    enhanced_tensor.index_copy_(0, group1_idx, group1)
    enhanced_tensor.index_copy_(0, group2_idx, group2)
    enhanced_tensor.index_copy_(0, group3_idx, group3)
    enhanced_tensor.index_copy_(0, group4_idx, group4)
    enhanced_tensor.index_copy_(0, group5_idx, group5)
    
    return enhanced_tensor

def mix_mask_obs(x,aug_x):
    N = x.size(0)
    enhanced_tensor = torch.empty_like(x).to(x.device)
    indices = torch.randperm(N).to(x.device)
    quarter = N // 3
    group1_idx = indices[:quarter]
    group2_idx = indices[quarter: 2 * quarter]
    group3_idx = indices[2 * quarter:]
    group1 = aug_x[group1_idx]
    group2 = x[group2_idx]
    group3 = x[group3_idx]
    enhanced_tensor.index_copy_(0, group1_idx, group1)
    enhanced_tensor.index_copy_(0, group2_idx, group2)
    enhanced_tensor.index_copy_(0, group3_idx, group3)
    return enhanced_tensor


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.repr_dim = 32 * 21 * 21
        self.layers = [
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
        ]
        for k in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))        
        self.layers = nn.Sequential(*self.layers)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = x / 255.0 - 0.5
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        return F.normalize(x,dim=-1)

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std,is_train=False):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        if not is_train:
            return dist
        else:
            return dist,mu
class Mask_org(nn.Module):
    def __init__(self,encoder,crtic):
        super().__init__()
        self.encoder=encoder
        self.critic=crtic
    def forward(self, obs, action):
        obs=self.encoder(obs)
        return self.critic(obs,action)
    


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True))
        self.w1=nn.Linear(hidden_dim, 1)
        self.w2=nn.Linear(hidden_dim, 1)
        self.apply(utils.weight_init)
    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1_reper = self.Q1(h_action)
        q2_reper = self.Q2(h_action)
        q1=self.w1(q1_reper)
        q2=self.w2(q2_reper)
        return q1, q2, q1_reper, q2_reper


class MuhaAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,gamma):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.gamma=gamma
        # models
        self.encoder = SharedCNN(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.beta_dis = torch.distributions.beta.Beta(1.0, 1.0)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.random_quant=RandomizedQuantizationAugModule(region_num=8)
        
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def drd_loss(self,rep,aug_rep):
        return torch.einsum('ij,ij->i', [F.normalize(rep,dim=-1), F.normalize(aug_rep,dim=-1)]).mean()
    
    def update_critic(self, obs, action, reward, discount, next_obs, step,aug_obs):
        metrics = dict()
        Q1, Q2, q1_reper, q2_reper = self.critic(obs, action)
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2, q1_reper_next, q2_reper_next = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)#+5e-5*rep_loss
        aug_Q1, aug_Q2, q1_aug_reper, q2_aug_reper = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        
        drd_loss_1=self.drd_loss(q1_reper,q1_reper_next)
        drd_loss_2=self.drd_loss(q2_reper,q2_reper_next)
        drd_loss3=self.drd_loss(q1_aug_reper,q1_reper_next)
        drd_loss4=self.drd_loss(q2_aug_reper,q2_reper_next)
        up_loss = drd_loss_1+drd_loss_2+drd_loss3+drd_loss4
        obs_loss = F.mse_loss(obs, aug_obs)
        norm_loss = 0.5*(1- torch.einsum('ij,ij->i', [obs, aug_obs])).mean()
        critic_loss = 0.5 * (aug_loss + critic_loss) + self.gamma*up_loss
 
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics
    def log_tensorboard(self, obs, action, step, writer, prefix="original"):
        if writer is None:
            return 
        model = Mask_org(self.encoder,self.critic)
        obs_grad = compute_attribution(model, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, quantile=0.9)
        grid = make_obs_grid(obs)
        writer.add_image(prefix + "/observation", grid, global_step=step)
        mask = (mask > 0.5).float()
        masked_obs = make_obs_grid(obs * mask)
        writer.add_image(prefix + "/masked_obs{}", masked_obs, global_step=step)

        for q in [0.5,0.7,0.8, 0.95, 0.975, 0.995]:
            mask = compute_attribution_mask(obs_grad, quantile=q)
            masked_obs = make_obs_grid(obs * mask)
            writer.add_image(
                prefix + "/attrib_q{}".format(q), masked_obs, global_step=step
            )
    
    def get_mask(self, obs, action, aug_obs):
        model = Mask_org(self.encoder,self.critic)
        obs_grad = compute_attribution(model, obs, action.detach())
        q=random.choice([0.9,0.925,0.95])
        mask = compute_attribution_mask(obs_grad, quantile=q)
        return self.aug(mask*aug_obs) 

 
 
    def update_actor(self, obs, step, mask_obs):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist,_ = self.actor(obs, stddev,is_train=True)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2,_,_ = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics
    
    
    def update(self, replay_iter, step, writer=None):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        with torch.no_grad():
            aug_obs = utils.random_overlay_repeat(original_obs,repeat_radio=0.6)
            mask_obs=self.get_mask(obs,action,aug_obs)
 
        next_obs = self.aug(next_obs.float())
        obs = self.encoder(obs)
        
        aug_obs = mix_mask_obs(aug_obs,mask_obs)
        aug_obs = random_aug(aug_obs,self.random_quant)

        aug_obs = self.encoder(aug_obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step,aug_obs))

        # update actor
        
        metrics.update(self.update_actor(obs.detach(), step, mask_obs.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
