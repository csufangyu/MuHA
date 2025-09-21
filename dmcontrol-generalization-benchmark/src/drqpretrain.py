# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import utils
import einops
from resmodel import ResNet
from resrgb import RGBBranch
from algorithms.rl_utils import compute_attribution,compute_attribution_mask,make_obs_grid
from torch.utils.tensorboard import SummaryWriter

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


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        # self.model = resnet18(pretrained=False)
        # model_path = '/root/autodl-tmp/models/resnet18-f37072fd.pth'
        # pretrained_dict = torch.load(model_path)
        # self.model.load_state_dict(pretrained_dict)
        restore_path="/root/autodl-tmp/models/KN_1M_resnet18.pth"
        self.model = ResNet(size=18, pretrained=None, restore_path=restore_path,
                          norm_cfg=dict(name='group_norm', num_groups=16))._model
        for param in self.model.parameters():
            param.requires_grad = False
        self.image_channel = 3     
    @torch.no_grad()
    def forward_conv(self, obs):
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs_list=[]
        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'relu':
                obs_list.append(obs)
            if name == 'layer2':
                obs_list.append(obs)
                break
        conv_list=[]
        for obs in obs_list:
            conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
            conv_prev = torch.cat([conv_current, conv_prev], axis=1)
            conv = conv_prev.view(conv_prev.size(0), conv_prev.size(1) * conv_prev.size(2), conv_prev.size(3), conv_prev.size(4))
            conv_list.append(conv)
        return conv_list


    def forward(self, obs):
        out_list = self.forward_conv(obs)
        return out_list

class ResRGB(nn.Module):
    def __init__(self):
        super(ResRGB, self).__init__()
        self.model=RGBBranch(arch="ResNet-18")
        model_path="/root/autodl-tmp/models/RGB_ResNet18_Places.pth.tar"
        pretrained_dict = torch.load(model_path)["state_dict"]
        self.model.load_state_dict(pretrained_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.image_channel = 3     
    @torch.no_grad()
    def forward_conv(self, obs):
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs_list=[]
        for name, module in self.model._modules.items():
            if name == 'in_block':
                obs,_ = module(obs)
            else:
                obs = module(obs)
            if name == 'encoder2':
                obs_list.append(obs)
                break
        conv_list=[]
        for obs in obs_list:
            conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
            conv_prev = torch.cat([conv_current, conv_prev], axis=1)
            conv = conv_prev.view(conv_prev.size(0), conv_prev.size(1) * conv_prev.size(2), conv_prev.size(3), conv_prev.size(4))
            conv_list.append(conv)
        return conv_list


    def forward(self, obs):
        out_list = self.forward_conv(obs)
        return out_list


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.out_dim = 16 * 35 * 35
        self.repr_dim = 1024
        self.pre_encoder=ResRGB()
        self.conv1= nn.Sequential(nn.Conv2d(obs_shape[0], 16, 3, stride=2),
                                     nn.ReLU())
        self.fc1 = nn.Linear(self.out_dim, self.repr_dim)
        self.ln1 = nn.LayerNorm(self.repr_dim)
        self.convnet = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU())
        
        self.fc2 = nn.Linear(61952, self.repr_dim)
        self.ln2 = nn.LayerNorm(self.repr_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        b = obs.size(0)
        pre_feature_list=self.pre_encoder(obs)
        fc2=self.ln2(self.fc2(pre_feature_list[-1].view(b,-1)))
        return fc2


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

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


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
        self.sw1=nn.Parameter(torch.randn(1))
        self.sw2=nn.Parameter(torch.randn(1))
        self.apply(utils.weight_init)
    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1_reper = F.normalize(self.Q1(h_action))
        q2_reper = F.normalize(self.Q2(h_action))
        q1=self.w1(q1_reper)*self.sw1
        q2=self.w2(q2_reper)*self.sw2
        return q1, q2,  q1_reper, q2_reper


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
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

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.writer = SummaryWriter("/root/autodl-tmp/drqpretrain_log/img")
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs_ = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        obs = self.encoder(obs_)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        # obs_grad = compute_attribution(self.encoder,self.critic, obs_.detach(), action.detach())
        # mask = compute_attribution_mask(obs_grad, quantile=0.90)
        # masked_obs = make_obs_grid(obs_ * mask)
        # for q in [0.95, 0.975, 0.9, 0.995, 0.999]:
        #     self.writer.add_image(
        #         "original" + "/attrib_q{}".format(q), masked_obs, global_step=step
        #     )
        return action.cpu().numpy()[0]

    def compute_drd(self,rep1,rep2,discount,reward,last_nnw):
        a = torch.einsum('ij,ij->i', [rep1, rep2])
        b = 1/discount
        c = reward**2/(2*discount*torch.norm(last_nnw)**2)
        return a-b+c
    def compute_drd_reward(self,rep1,rep2):
        reward = 1 - torch.einsum('ij,ij->i', [rep1, rep2])
        return reward.view(-1,1)
    def drd_loss(self,rep, aug_rep):
        return (1- torch.einsum('ij,ij->i', [rep.detach(), aug_rep])).abs().mean()
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
        drd_loss=self.drd_loss(q1_reper,q1_aug_reper)+self.drd_loss(q2_reper,q2_aug_reper)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        critic_loss = 0.5 * (critic_loss + aug_loss) #+0.0001*drd_loss
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['drd'] = drd_loss.mean().item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2,_,_ = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        return metrics
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        # strong augmentation
        aug_obs = self.encoder(utils.random_overlay(original_obs))

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step,aug_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
