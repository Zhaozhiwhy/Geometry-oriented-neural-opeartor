# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:08:19 2024

@author: zzw
"""
from layers import *
from models import *
import csv
import torch as t
from Loss_function import LpLoss, count_params,MixedLoss
from Adam import Adam
from scipy.spatial import Delaunay, ConvexHull
import boundry_surface
from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator
from matplotlib.colors import Normalize,BoundaryNorm, ListedColormap
import matplotlib.tri as tri
from scipy.interpolate import griddata
from torch import nn, einsum
from torch.cuda.amp import GradScaler, autocast
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def readfile(path):
	f = open(path)
	rows = list(csv.reader(f))
	f.close()
	return rows


def openfile(filename):
	K = readfile(filename)
	for t in range(len(K)):
		K[t] = [float(V) for V in K[t]]
	return K


class Mesh_loader2(object):
	def __init__(self, data_location, set_type):

		self.data_location = data_location
		self.names = self.load_names()

		self.position = []
		self.edge = []
		self.a = []
		self.u = []
		self.knn_idx = []

		sample_number = range(len(self.names))
		self.sample_number = len(self.names)

		self.deformation_shape_list = []
		for i in tqdm(sample_number):
			obj = self.names[i]

			number = int(obj[0:-4])

			position = np.load(data_location + '/geometry/' + str(number) + '.npy')
			stress_points = np.load(data_location + '/stress/' + str(number) + '.npy')
			deformation_points = np.load(data_location + '/deformation/' + str(number) + '.npy')
			edge = np.load(data_location + '/edges/' + str(number) + '.npy')
			self.deformation_shape_list.append(deformation_points.shape[0])
			position = position.reshape(1, -1, 3)
			position = torch.FloatTensor(position)

			stress_points = stress_points[np.random.choice(stress_points.shape[0], 2000, replace=False)]
			deformation_points = self.weighted_sampling(deformation_points, num_samples=10000)

			self.position.append(position)
			self.a.append(stress_points)
			self.u.append(deformation_points)
			self.edge.append(edge)
		print("self.deformation_shape_list max:", np.array(self.deformation_shape_list).min(),
			  self.names[np.array(self.deformation_shape_list).argmin()])

	def load_names(self):
		# List to hold the names of all .obj files
		return [file for file in os.listdir(self.data_location + 'deformation') if file.endswith('.npy')]

	def __len__(self):

		return self.sample_number

	def weighted_sampling(self, deformation_points, num_samples):

		xyz = deformation_points[:, :3]


		x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
		x_center = (x_max + x_min) / 2


		distances_from_center = np.abs(xyz[:, 0] - x_center)
		weights = distances_from_center / np.max(distances_from_center)  # ?????

		sampled_indices = np.random.choice(len(deformation_points), size=num_samples, replace=False,
										   p=weights / np.sum(weights))
		return deformation_points[sampled_indices]

	def __getitem__(self, index):

		data = {}


		data['position'] = self.position[index]

		data['a'] = self.a[index]  # [np.random.permutation(self.a[index].shape[0])[:10000]]
		data['u'] = self.u[index]  # [np.random.permutation(self.u[index].shape[0])[:20000]]
		data['name'] = self.names[index]
		data['edge'] = self.edge[index]
		return data

	def collate(self, batch):

		data = {}

		data['positions'] = [item['position'] for item in batch]
		data['edges'] = [item['edge'] for item in batch]
		data['as'] = [item['a'] for item in batch]

		data['us'] = [item['u'] for item in batch]
		data['names'] = [item['name'] for item in batch]
		return data



class ResidualBlock(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(ResidualBlock, self).__init__()
		self.fc1 = nn.Linear(input_dim, output_dim)
		self.fc2 = nn.Linear(output_dim, output_dim)
		self.prelu1 = nn.PReLU()
		self.prelu2 = nn.PReLU()
		self.shortcut = nn.Sequential()

		if input_dim != output_dim:
			self.shortcut = nn.Sequential(
				nn.Linear(input_dim, output_dim)
			)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		identity = self.shortcut(x)
		out = self.prelu1(self.fc1(x))
		out = self.fc2(out)
		out += identity  # ????
		out = self.prelu2(out)
		return out


class XYtoZParams(nn.Module):
	def __init__(self, g_dim, z_dim):
		super(XYtoZParams, self).__init__()
		self.g_dim = g_dim
		self.z_dim = z_dim

		self.res_block1 = ResidualBlock(2+128+ self.g_dim, 128)
		self.res_block2 = ResidualBlock(128, 128)

		self.coord_embed1 = nn.Linear(3 + self.g_dim, 128)
		self.coord_embed2 = nn.Linear(3 + self.g_dim, 128)

		self.coors = CoordinateEmbedding(3,128)
		self.fc_out = nn.Linear(128, self.z_dim)
		self.cross_attention = nn.MultiheadAttention(embed_dim=self.z_dim, num_heads=8)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def aggregate(self, r):
		return torch.mean(r, dim=1)

	def cross_attention_aggregate(self, geometry_feat, coords, point_features):

		query = geometry_feat.unsqueeze(0)  #  (1, batch_size, z_dim)
		key_value = coords.permute(1, 0, 2)  # (seq_len, batch_size, z_dim)
		v_value = point_features.permute(1, 0, 2)
		cross_attn_output, _ = self.cross_attention(query, key_value, v_value)  # ???????

		return cross_attn_output.squeeze(0)  # (batch_size, z_dim)

	def forward(self, x, y, mesh_context):
		mesh_context = mesh_context.unsqueeze(0).repeat(1, x.shape[1], 1)
		x1 = self.coors(x)
		x_y = torch.cat([x1, y,mesh_context], dim=-1)

		coords = torch.cat([x, mesh_context], dim=-1)

		x_y = self.res_block1(x_y)
		x_y += self.coord_embed1(coords)  # ??????????

		# ???????????
		x_y = self.res_block2(x_y)
		x_y += self.coord_embed2(coords)  # ??????????

		r_i = self.fc_out(x_y)

		r = self.aggregate(r_i)

		return r

class CoordinateEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CoordinateEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.prelu(x)
        return x

class QNetwork(nn.Module):
	def __init__(self, g_dim, z_dim):
		super(QNetwork, self).__init__()
		self.g_dim = g_dim
		self.z_dim = z_dim

		self.res_block1 = ResidualBlock(self.g_dim + self.z_dim + 128, 256)
		self.res_block2 = ResidualBlock(256, 128)

		self.fc_out = nn.Linear(128, 1)
		self.coords  = CoordinateEmbedding(3,128)
		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, z_sample, x_target, mesh_context):
		mesh_context = mesh_context.unsqueeze(0).repeat(1, x_target.shape[1], 1)
		x_target = self.coords(x_target)
		z_sample = z_sample.unsqueeze(1).expand(-1, x_target.shape[1], -1)
		z_x = torch.cat([z_sample, x_target, mesh_context], dim=-1)
		out = self.res_block1(z_x)
		out = self.res_block2(out)

		y_hat = self.fc_out(out)
		return y_hat


class NP1(nn.Module):
	def __init__(self, r_dim, z_dim):
		super(NP1, self).__init__()
		self.z_dim = 128
		self.g_dim = 128

		# ??????
		self.encoder_mesh = MeshEncoder(self.g_dim)
		self.xy_to_z_params = XYtoZParams(self.g_dim, self.z_dim)
		self.q_network = QNetwork(self.g_dim, self.z_dim)

	def forward(self, x_context, y_context, x_target, position, edges):
		position = position.reshape(-1, 3)
		feat = self.encoder_mesh(position, edges)
		z_context = self.xy_to_z_params(x_context, y_context, feat)
		y_hat = self.q_network(z_context, x_target, feat)

		assert torch.isnan(position).sum() == 0, print('position:',position)
		assert torch.isnan(edges).sum() == 0, print('edges:',edges)
		assert torch.isnan(feat).sum() == 0, print('feat:',feat)
		assert torch.isnan(z_context).sum() == 0, print('z_context:',z_context)
		assert torch.isnan(y_hat).sum() == 0, print('y_hat:',y_hat)

		return y_hat


if __name__ == '__main__':

	train_dataset = 'data_to'
	test_dataset = 'data_to'

	save_result_path = './result_' + train_dataset + '/' + test_dataset

	path_model = 'model_mesh_3d_gnn_attn_rel2_beam_frame_' + train_dataset
	imgpath = 'loss/' + path_model
	path_model = 'model/' + path_model
	if not os.path.exists(save_result_path):
		if not os.path.exists('./result_' + train_dataset):
			os.mkdir('./result_' + train_dataset)
		os.mkdir(save_result_path)

	au_train = './frame_beam_noise_train/'
	au_test = './beam_frame_test/'

	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	batch_size = 8
	learning_rate = 0.0003
	scheduler_step = 400
	scheduler_gamma = 0.99
	epochs = 6000
	geometry_scale = 600
	stress_scale = 100

	test_data = Mesh_loader2(au_test, set_type='test')
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_data.collate)
	train_data = Mesh_loader2(au_train, set_type='train')
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
							  collate_fn=train_data.collate, pin_memory=False)

	if_train = True
	# if_train = False
	current_time = datetime.now().strftime('%Y%m%d%H%M')
	scaler = GradScaler()
	writer = SummaryWriter('runs/grad_check')
	if if_train == True:
		model = NP1(128, 128).to(device)
# 		model = torch.load(path_model, map_location='cuda:0').to(device)
		optimizer = torch.optim.AdamW([
			{'params': model.encoder_mesh.parameters(), 'lr': 1e-3},
			{'params': model.q_network.parameters(), 'lr': 1e-4},
			{'params': model.xy_to_z_params.parameters(), 'lr': 1e-4}
		], lr=1e-4,weight_decay=5e-6,betas=(0.9, 0.999)) 

		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

		myloss = MixedLoss(alpha =1,size_average=True)

		step = 0
		model.train()
		best_loss = 10
		for ep in range(0, 6000):

			loss_train = []
			loss_test = []
			start = time.time()
			for batch in train_loader:
				count = 0
				pre_y = None
				for position, a, u, edge in zip(batch['positions'], batch['as'], batch['us'], batch['edges']):

					position = position.to(device) / geometry_scale
					edge = torch.FloatTensor(edge).to(device)
					a = torch.FloatTensor(a).to(device)
					u = torch.FloatTensor(u).to(device)
					context_x = a[:, 0:-2].reshape(1, -1, 3).to(device) / geometry_scale
					context_y = a[:, -2:].reshape(1, -1, 2).to(device) / stress_scale
					target_x = u[:, 0:-1].reshape(1, -1, 3).to(device) / geometry_scale
					target_y = u[:, -1].reshape(1, -1, 1).to(device)

					assert torch.isnan(context_x).sum() == 0, print('context_x:')
					assert torch.isnan(context_y).sum() == 0, print('context_y:')
					assert torch.isnan(target_x).sum() == 0, print('target_x:')
					assert torch.isnan(target_y).sum() == 0, print('target_y:')
					rotate = random.randint(1, 10)
					if rotate <= 5:
						position[:, :, 0] = -position[:, :, 0]
						context_x[:, :, 0] = -context_x[:, :, 0]
						target_x[:, :, 0] = -target_x[:, :, 0]
					rotate = random.randint(1, 10)
					if rotate <= 5:
						position[:, :, 1] = -position[:, :, 1]
						context_x[:, :, 1] = -context_x[:, :, 1]
						target_x[:, :, 1] = -target_x[:, :, 1]
					movement = random.randint(-10, 10)
					if movement <= 5:
					
						position[:, :, 1] = -position[:, :, 1]
						context_x[:, :, 1] = -context_x[:, :, 1]
						target_x[:, :, 1] = -target_x[:, :, 1]

					with autocast():
						if pre_y is None:
							pre_y = model(context_x, context_y, target_x, position, edge)
						else:
							pre_y = torch.cat((pre_y, model(context_x, context_y, target_x, position, edge)))

					count = count + 1
				
				target_y_log = (torch.FloatTensor(batch['us'])[:, :, -1].reshape(count, -1).float()).to(device)

				pred_y_shifted = pre_y + 0.005 * torch.sign(pre_y)
				target_y_log = target_y_log + 0.005 * torch.sign(target_y_log)
				
				loss_y_train = myloss(pred_y_shifted.reshape(count, -1).float(),
									target_y_log)
				
				assert torch.isnan(pre_y).sum() == 0, print('nan_pre_y:')
				assert torch.isnan(loss_y_train).sum() == 0, print('nan_loss:')

				loss_train.append(loss_y_train.item())
				optimizer.zero_grad()
				loss_y_train.backward()

				optimizer.step()

				step = step + 1
				if step % 10 == 0:

					end = time.time()
					with torch.no_grad():
						for batch in test_loader:
							count = 0
							pre_y = None
							for position, a, u, edge in zip(batch['positions'], batch['as'], batch['us'],
															batch['edges']):

								position = position.to(device) / geometry_scale
								edge = torch.FloatTensor(edge).to(device)
								a = torch.FloatTensor(a).to(device)
								u = torch.FloatTensor(u).to(device)
								context_x = a[:, 0:-2].reshape(1, -1, 3).to(device) / geometry_scale
								context_y = a[:, -2:].reshape(1, -1, 2).to(device) / stress_scale
								target_x = u[:, 0:-1].reshape(1, -1, 3).to(device) / geometry_scale

								with autocast():
									if pre_y is None:
										pre_y = model(context_x, context_y, target_x, position, edge)
									else:
										pre_y = torch.cat(
											(pre_y, model(context_x, context_y, target_x, position, edge)))
								count = count + 1
								torch.cuda.empty_cache()

							target_y_log = torch.FloatTensor(np.array(batch['us'])[:, :, -1]).reshape(count,-1).float().to(device)
							pred_y_shifted = pre_y + 0.05 * torch.sign(pre_y)
							target_y_log = target_y_log + 0.05 * torch.sign(target_y_log)
							
							loss_y_test = myloss(pred_y_shifted.reshape(count, -1).float(),
												target_y_log)

							loss_test.append(loss_y_test.item())

						eror = pre_y.reshape(count, -1)- torch.FloatTensor(np.array(batch['us'])[:, :, -1]).reshape(
							count, -1).to(
							device)

						print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
						print('step:', step, ' \ntrain_loss:', np.array(loss_train).mean(), ' \ntest_loss:',
							  np.array(loss_test).mean(),'\nmax_a_error: ',abs(eror).max().item(), ' \ntime: ', end - start)
						if np.array(loss_train).mean() < best_loss:
							torch.save(model, path_model)
							best_loss = np.array(loss_train).mean()
						with open('./loss/train_loss_gnn_beam_frame' + current_time + '.csv', 'a+') as f:
							np.savetxt(f, np.array(loss_train).mean().reshape(1, 1), delimiter=',')
						with open('./loss/test_loss_gnn_beam_frame' + current_time + '.csv', 'a+') as f:
							np.savetxt(f, np.array(loss_test).mean().reshape(1, 1), delimiter=',')
						loss_train = []
						loss_test = []

				torch.cuda.empty_cache()
			scheduler.step()