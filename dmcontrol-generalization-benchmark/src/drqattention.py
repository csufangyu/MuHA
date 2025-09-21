# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
import torchvision
from torchvision import transforms
import utils
import os
import torchvision.transforms as TF
import torchvision.datasets as datasets
import json
import einops
def load_config(key=None):
	path = os.path.join('/root/PIE-G/cfgs', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data
places_dataloader = None
places_iter = None

def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


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
        self.model = resnet18(pretrained=False)
        model_path = '/root/autodl-tmp/models/resnet18-f37072fd.pth'
        pretrained_dict = torch.load(model_path)
        self.model.load_state_dict(pretrained_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
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
            # conv_prev = torch.cat([conv_current, conv_prev], axis=1)
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
        self.pre_encoder=ResEncoder()
        for param in self.pre_encoder.parameters():
            param.requires_grad = False
        self.conv1= nn.Sequential(nn.Conv2d(obs_shape[0], 16, 3, stride=2),
                                     nn.ReLU())
        self.fc1 = nn.Linear(self.out_dim, self.repr_dim//2)
        self.ln1 = nn.LayerNorm(self.repr_dim//2)
        self.convnet = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(16, 16, 3, stride=1),
                                     nn.ReLU())
        self.fc2 = nn.Linear(30976, self.repr_dim//2)
        self.ln2 = nn.LayerNorm(self.repr_dim//2)
        self.atten = CrossAttention(self.repr_dim,self.repr_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        b = obs.size(0)
        pre_feature_list=self.pre_encoder(obs)
        fc2=self.ln2(self.fc2(pre_feature_list[1].view(b,-1)))
        h = self.convnet(self.conv1(obs)).view(b,-1)
        h = self.fc1(h)
        h = self.ln1(h)
        h = torch.cat([h,fc2],dim=1)
        out = self.atten(h,h)
        return out
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.ln=nn.LayerNorm(query_dim)
    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (n h)  -> b n h ', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = einops.rearrange(out, 'b n h -> b (n h)', h=h)
        out=self.to_out(out) + context
        return self.ln(out)


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
    def __init__(self,obs_shape, action_shape, args):
    # def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
    #              hidden_dim, critic_target_tau, num_expl_steps,
    #              update_every_steps, stddev_schedule, stddev_clip, use_tb):
        device="cuda:0"
        critic_target_tau=0.01
        update_every_steps= 2
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = True
        self.num_expl_steps = 2000
        self.stddev_schedule = "linear(1.0,0.1,100000)"
        self.stddev_clip = 0.3
        feature_dim=256
        hidden_dim=1024
        lr=0.0001
        # models
        self.encoder = Encoder().to(device)
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
            inter_reward = torch.min(self.compute_drd_reward(q1_reper,q1_reper_next),
                                     self.compute_drd_reward(q2_reper,q2_reper_next))
            target_Q = target_Q + 0.0001*inter_reward

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)#+5e-5*rep_loss
        aug_Q1, aug_Q2, q1_aug_reper, q2_aug_reper = self.critic(aug_obs, action)
        drd_loss=self.drd_loss(q1_reper,q1_aug_reper)+self.drd_loss(q2_reper,q2_aug_reper)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)+0.00001*drd_loss
        with torch.no_grad():
            DRD=self.compute_drd(q1_reper, q1_reper_next,discount,reward,self.critic.w1.weight*self.critic.sw1)+\
                self.compute_drd(q2_reper,q2_reper_next,discount,reward,self.critic.w2.weight*self.critic.sw2)
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['drd'] = DRD.mean().item()

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
