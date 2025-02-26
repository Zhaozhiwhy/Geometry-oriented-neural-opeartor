import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import *
import torchvision.models as models


class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True))

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True))

		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True))

		self.layer4 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True))

		self.layer5 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True))

		self.layer6 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True))

		self.layer7 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True))

		self.layer8 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True))

		self.layer9 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True))

		self.layer10 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True))

		self.layer11 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True))

		self.layer12 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True))

		self.layer13 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True))

		self.layer14 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True))

		self.layer15 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True))

		self.layer16 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True))

		self.layer17 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True))
		self.layer18 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True))

	def forward(self, tensor):
		x = self.layer1(tensor)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		A = x
		x = self.layer9(x)
		x = self.layer10(x)
		x = self.layer11(x)
		B = x
		x = self.layer12(x)
		x = self.layer13(x)
		x = self.layer14(x)
		C = x
		x = self.layer15(x)
		x = self.layer16(x)
		x = self.layer17(x)
		D = self.layer18(x)

		return A, B, C, D


class MeshDeformationBlock(nn.Module):
	def __init__(self, input_features, hidden=192, output_features=3):
		super(MeshDeformationBlock, self).__init__()
		self.gc1 = Image_ZERON_GCNGCN(input_features, hidden)
		self.gc2 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc3 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc4 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc5 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc6 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc7 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc8 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc9 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc10 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc11 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc12 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc13 = Image_ZERON_GCNGCN(hidden, hidden)
		self.gc15 = Image_ZERON_GCNGCN(hidden, output_features)
		self.hidden = hidden

	def forward(self, features, pooled, adj):
		full_features = torch.cat((features, pooled), dim=1)
		# 1
		x = (self.gc1(full_features, adj, F.relu))
		x = (self.gc2(x, adj, F.relu))
		x = full_features[:, :self.hidden] + x
		features = x
		features /= 2

		# 2
		x = (self.gc3(features, adj, F.relu))
		x = (self.gc4(x, adj, F.relu))
		features = features + x
		features /= 2
		# 3
		x = (self.gc5(features, adj, F.relu))
		x = (self.gc6(x, adj, F.relu))
		features = features + x
		features /= 2

		# 4
		x = (self.gc7(features, adj, F.relu))
		x = (self.gc8(x, adj, F.relu))
		features = features + x
		features /= 2

		# 5
		x = (self.gc9(features, adj, F.relu))
		x = (self.gc10(x, adj, F.relu))
		features = features + x
		features /= 2

		# 6
		x = (self.gc11(features, adj, F.relu))
		x = (self.gc12(x, adj, F.relu))
		features = features + x
		features /= 2

		# 7
		x = (self.gc13(features, adj, F.relu))

		features = features + x
		features /= 2

		coords = (self.gc15(features, adj, lambda x: x))
		return features, coords


class BatchMeshDeformationBlock(nn.Module):
	def __init__(self, input_features, verts, hidden=192, output_features=3):
		super(BatchMeshDeformationBlock, self).__init__()
		self.gc1 = Batch_Image_ZERON_GCNGCN(input_features, hidden)
		self.gc2 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc3 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc4 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc5 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc6 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc7 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc8 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc9 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc10 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc11 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc12 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc13 = Batch_Image_ZERON_GCNGCN(hidden, hidden)
		self.gc15 = Batch_Image_ZERON_GCNGCN(hidden, output_features)
		self.hidden = hidden

		self.bn1 = nn.BatchNorm1d(verts)
		self.bn2 = nn.BatchNorm1d(verts)
		self.bn3 = nn.BatchNorm1d(verts)
		self.bn4 = nn.BatchNorm1d(verts)
		self.bn5 = nn.BatchNorm1d(verts)
		self.bn6 = nn.BatchNorm1d(verts)
		self.bn7 = nn.BatchNorm1d(verts)
		self.bn8 = nn.BatchNorm1d(verts)
		self.bn9 = nn.BatchNorm1d(verts)
		self.bn10 = nn.BatchNorm1d(verts)
		self.bn11 = nn.BatchNorm1d(verts)
		self.bn12 = nn.BatchNorm1d(verts)
		self.bn13 = nn.BatchNorm1d(verts)
		self.bn14 = nn.BatchNorm1d(verts)

	def forward(self, features, pooled, adj):
		full_features = torch.cat((features, pooled), dim=-1)

		x = (self.gc1(full_features, adj, lambda x: x))
		x = F.relu(self.bn1(x))
		x = (self.gc2(x, adj, lambda x: x))
		x = F.relu(self.bn2(x))

		x = full_features[:, :, :self.hidden] + x
		features = x
		features /= 2

		# 2
		x = (self.gc3(features, adj, lambda x: x))
		x = F.relu(self.bn3(x))
		x = (self.gc4(x, adj, lambda x: x))
		x = F.relu(self.bn4(x))
		features = features + x
		features /= 2
		# 3
		x = (self.gc5(features, adj, lambda x: x))
		x = F.relu(self.bn5(x))
		x = (self.gc6(x, adj, lambda x: x))
		x = F.relu(self.bn6(x))
		features = features + x
		features /= 2

		# 4
		x = (self.gc7(features, adj, lambda x: x))
		x = F.relu(self.bn7(x))
		x = (self.gc8(x, adj, lambda x: x))
		x = F.relu(self.bn8(x))
		features = features + x
		features /= 2

		# 5
		x = (self.gc9(features, adj, lambda x: x))
		x = F.relu(self.bn9(x))
		x = (self.gc10(x, adj, lambda x: x))
		x = F.relu(self.bn10(x))
		features = features + x
		features /= 2

		# 6
		x = (self.gc11(features, adj, lambda x: x))
		x = F.relu(self.bn11(x))
		x = (self.gc12(x, adj, lambda x: x))
		x = F.relu(self.bn12(x))
		features = features + x
		features /= 2

		# 7
		x = (self.gc13(features, adj, lambda x: x))
		x = F.relu(self.bn13(x))

		features = features + x
		features /= 2

		coords = (self.gc15(features, adj, lambda x: x))
		return features, coords


class MeshEncoder(nn.Module):
	def __init__(self, latent_length):
		super(MeshEncoder, self).__init__()
		self.h1 = ZERON_GCN_edge(3, 128)
		self.h21 = ZERON_GCN_edge(128, 128)
		self.h22 = ZERON_GCN_edge(128, 128)
		self.h23 = ZERON_GCN_edge(128, 128)
		self.h24 = ZERON_GCN_edge(128, 128)
		self.h3 = ZERON_GCN_edge(128, 128)
		self.h4 = ZERON_GCN_edge(128, 128)
		self.h41 = ZERON_GCN_edge(128, 150)
		self.h5 = ZERON_GCN_edge(150, 300)
		# 		self.h6 = ZERON_GCN(200, 210)
		# 		self.h7 = ZERON_GCN(210, 250)
		# 		self.h8 = ZERON_GCN(250, 300)
		# 		self.h81 = ZERON_GCN(300, 300)
		# 		self.h9 = ZERON_GCN(300, 300)
		# 		self.h10 = ZERON_GCN(300, 300)
		# 		self.h11 = ZERON_GCN(300, 300)
		self.reduce = GCNMax_edge(300, latent_length)

	def resnet(self, features, res):
		temp = features[:, :res.shape[1]]
		temp = temp + res
		features = torch.cat((temp, features[:, res.shape[1]:]), dim=1)
		return features, features

	def forward(self, positions, adj, play=False):
		# print positions[:5, :5]
		res = positions
		features = self.h1(positions, adj, F.elu)
		# assert torch.isnan(features).sum() == 0, print('1:')
		features = self.h21(features, adj, F.elu)
		# assert torch.isnan(features).sum() == 0, print('2:')
		features = self.h22(features, adj, F.elu)
		# assert torch.isnan(features).sum() == 0, print('3:')
		features = self.h23(features, adj, F.elu)
		# assert torch.isnan(features).sum() == 0, print('4:')
		features = self.h24(features, adj, F.elu)
		# print(features.max(),features.min())
		# assert torch.isnan(features).sum() == 0, print('5:')
		features = self.h3(features, adj, F.elu)
		features = self.h4(features, adj, F.elu)
		features = self.h41(features, adj, F.elu)
		features = self.h5(features, adj, F.elu)
		# 		features = self.h6(features, adj, F.elu)
		# 		features = self.h7(features, adj, F.elu)
		# 		features = self.h8(features, adj, F.elu)
		# 		features = self.h81(features, adj, F.elu)
		# 		features = self.h9(features, adj, F.elu)
		# 		features = self.h10(features, adj, F.elu)
		# 		features = self.h11(features, adj, F.elu)

		latent = self.reduce(features, adj, F.elu)
		assert torch.isnan(latent).sum() == 0, print('5:')

		return latent


class Decoder(nn.Module):
	def __init__(self, latent_length):
		super(Decoder, self).__init__()
		self.fully = torch.nn.Sequential(
			torch.nn.Linear(latent_length, 512)
		)

		self.model = torch.nn.Sequential(
			torch.nn.ConvTranspose3d(64, 64, 4, stride=2, padding=(1, 1, 1), ),
			nn.BatchNorm3d(64),
			nn.ELU(inplace=True),

			torch.nn.ConvTranspose3d(64, 64, 4, stride=2, padding=(1, 1, 1)),
			nn.BatchNorm3d(64),
			nn.ELU(inplace=True),

			torch.nn.ConvTranspose3d(64, 32, 4, stride=2, padding=(1, 1, 1)),
			nn.BatchNorm3d(32),
			nn.ELU(inplace=True),

			torch.nn.ConvTranspose3d(32, 8, 4, stride=2, padding=(1, 1, 1)),
			nn.BatchNorm3d(8),
			nn.ELU(inplace=True),

			nn.Conv3d(8, 1, (3, 3, 3), stride=1, padding=(1, 1, 1))
		)

	def forward(self, latent):
		decode = self.fully(latent).view(-1, 64, 2, 2, 2)
		decode = self.model(decode).reshape(-1, 32, 32, 32)
		voxels = F.sigmoid(decode)
		return voxels


class BatchMeshEncoder(nn.Module):
	def __init__(self, latent_length):
		super(BatchMeshEncoder, self).__init__()
		self.h1 = BatchZERON_GCN(3, 60)
		self.h21 = BatchZERON_GCN(60, 60)
		self.h22 = BatchZERON_GCN(60, 60)
		self.h23 = BatchZERON_GCN(60, 60)
		self.h24 = BatchZERON_GCN(60, 120)
		self.h3 = BatchZERON_GCN(120, 120)
		self.h4 = BatchZERON_GCN(120, 120)
		self.h41 = BatchZERON_GCN(120, 150)
		self.h5 = BatchZERON_GCN(150, 200)
		self.h6 = BatchZERON_GCN(200, 210)
		self.h7 = BatchZERON_GCN(210, 250)
		self.h8 = BatchZERON_GCN(250, 300)
		self.h81 = BatchZERON_GCN(300, 300)
		self.h9 = BatchZERON_GCN(300, 300)
		self.h10 = BatchZERON_GCN(300, 300)
		self.h11 = BatchZERON_GCN(300, 300)
		self.reduce = BatchGCNMax(300, latent_length)

	def resnet(self, features, res):
		temp = features[:, :res.shape[1]]
		temp = temp + res
		features = torch.cat((temp, features[:, res.shape[1]:]), dim=1)
		return features, features

	def forward(self, positions, adj, play=False):
		# print positions[:5, :5]
		res = positions
		features = self.h1(positions, adj, F.elu)
		features = self.h21(features, adj, F.elu)
		features = self.h22(features, adj, F.elu)
		features = self.h23(features, adj, F.elu)
		features = self.h24(features, adj, F.elu)
		features = self.h3(features, adj, F.elu)
		features = self.h4(features, adj, F.elu)
		features = self.h41(features, adj, F.elu)
		features = self.h5(features, adj, F.elu)
		features = self.h6(features, adj, F.elu)
		features = self.h7(features, adj, F.elu)
		features = self.h8(features, adj, F.elu)
		features = self.h81(features, adj, F.elu)
		features = self.h9(features, adj, F.elu)
		features = self.h10(features, adj, F.elu)
		features = self.h11(features, adj, F.elu)

		latent = self.reduce(features, adj, F.elu)

		return latent
