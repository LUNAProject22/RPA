import torch
from torch import nn
import math
from torch.nn import functional as F
from timm.models.layers import DropPath

class MAP_Adapter(nn.Module):
	"MAP Adatper refered from OpenCLIP Github: " 
	"https://github.com/mlfoundations/open_clip/blob/d7a5a9595d68287e8ab24797df04d9a79d37faef/src/open_clip/transformer.py#L87"
	def __init__(
            self,
            dim,
            num_heads=1,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
	    	ratio=1.0/8,
		    skip_connect=True,
		    grid_h=0,
		    grid_w=0,
		    qk_type=0,
		    drop_path: float=0.0,
 	   ):
		super().__init__()

		self.scaled_cosine = scaled_cosine
		self.scale_heads = scale_heads
		assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.logit_scale_max = logit_scale_max
		self.skip_connect = skip_connect
		self.grid_h = grid_h
		self.grid_w = grid_w
		self.qk_type = qk_type

		self.rd_dim = int(dim * ratio)
		print("MAP_Adapter rd_dim: {}, ratio {}".format(self.rd_dim, ratio))
		# keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
		#randn = 0.01 * torch.randn((self.rd_dim * 3, dim))
		self.in_proj_weight = nn.Parameter(0.01 * torch.randn((self.rd_dim * 2, dim)) * self.scale)
		#self.in_proj_weight = nn.Parameter(randn[:self.rd_dim*2, :] * self.scale)
		if qkv_bias:
			self.in_proj_bias = nn.Parameter(torch.zeros(self.rd_dim * 2))
		else:
			self.in_proj_bias = None

		if self.scaled_cosine:
			self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
		else:
			self.logit_scale = None

		self.attn_drop = nn.Dropout(attn_drop)
		if self.scale_heads:
			self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
		else:
			self.head_scale = None
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


	def forward(self, x, attn_mask=None):
		L, N, C = x.shape
		#q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)		
		q, k = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(2, dim=-1)		
		q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
		k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
		
		if self.logit_scale is not None:
			attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
			logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
			attn = attn.view(N, self.num_heads, L, L) * logit_scale
			attn = attn.view(-1, L, L)
		else:
			q = q * self.scale
			attn = torch.bmm(q, k.transpose(-1, -2))

		attn = self.drop_path(attn)
		return attn

class Adapter(nn.Module):
	"Standard Adatper from AIM Github: " 
	"https://github.com/taoyang1122/adapt-image-models/blob/main/mmaction/models/backbones/vit_clip.py "
	def __init__(self, D_features, mlp_ratio=0.25, out_dim=0, act_layer=nn.GELU, skip_connect=True, drop_path: float=0.0):
		super().__init__()
		self.skip_connect = skip_connect
		D_hidden_features = int(D_features * mlp_ratio)
		self.act = act_layer()
		self.D_fc1 = nn.Linear(D_features, D_hidden_features)
		if out_dim == 0:
			self.D_fc2 = nn.Linear(D_hidden_features, D_features)
		else:
			self.D_fc2 = nn.Linear(D_hidden_features, out_dim)
		print("Adapter reduce rate {}".format(mlp_ratio))
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		# x is (HW+1, B, D)
		xs = self.D_fc1(x)
		xs = self.act(xs)
		xs = self.D_fc2(xs)
		if self.skip_connect:
			#print("DEBUG: Adapter Skip Branch")
			x = x + self.drop_path(xs)
		else:
			x = self.drop_path(xs)
		return x

class MSHA_Adapter(nn.Module):
	"Attention Adatper refered from OpenCLIP Github: " 
	"https://github.com/mlfoundations/open_clip/blob/d7a5a9595d68287e8ab24797df04d9a79d37faef/src/open_clip/transformer.py#L87"
	def __init__(
            self,
            dim,
            num_heads=1,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
	    	ratio=0.125,
		    skip_connect=True,
 	   ):
		super().__init__()
		self.scaled_cosine = scaled_cosine
		self.scale_heads = scale_heads
		assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.logit_scale_max = logit_scale_max
		self.skip_connect = skip_connect

		self.rd_dim = int(dim * ratio)
		print("MSHA_Adapter rd_dim: {}, ratio {}".format(self.rd_dim, ratio))
		# keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
		self.in_proj_weight = nn.Parameter(0.01 * torch.randn((self.rd_dim * 3, dim)) * self.scale)
		#self.q_proj = nn.Linear(dim, self.rd_dim)
		#self.k_proj = nn.Linear(dim, self.rd_dim)
		#self.v_proj = nn.Linear(dim, self.rd_dim)
		if qkv_bias:
			self.in_proj_bias = nn.Parameter(torch.zeros(self.rd_dim * 3))
		else:
			self.in_proj_bias = None

		if self.scaled_cosine:
			self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
		else:
			self.logit_scale = None

		self.attn_drop = nn.Dropout(attn_drop)
		if self.scale_heads:
			self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
		else:
			self.head_scale = None

		self.adapter_out_proj = nn.Linear(self.rd_dim, dim)
		self.out_drop   = nn.Dropout(proj_drop)
		


	def forward(self, x, attn_mask=None):
		L, N, C = x.shape
		q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)		
		#q = self.q_proj(x)
		#k = self.k_proj(x)
		#v = self.v_proj(x)
		q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1).contiguous()
		k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1).contiguous()
		v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1).contiguous()
		
		if self.logit_scale is not None:
			attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2).contiguous())
			logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
			attn = attn.view(N, self.num_heads, L, L) * logit_scale
			attn = attn.view(-1, L, L).contiguous()
		else:
			q = q * self.scale
			attn = torch.bmm(q, k.transpose(-1, -2).contiguous())

		if attn_mask is not None:
			if attn_mask.dtype == torch.bool:
				new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
				new_attn_mask.masked_fill_(attn_mask, float("-inf"))
				attn_mask = new_attn_mask
			attn += attn_mask

		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		xs = torch.bmm(attn, v)

		if self.head_scale is not None:
			xs = xs.view(N, self.num_heads, L, self.rd_dim).contiguous() * self.head_scale
			xs = xs.view(N, L, self.rd_dim).contiguous()
		xs = xs.transpose(0, 1).contiguous().reshape(L, N, self.rd_dim)
		xs = self.adapter_out_proj(xs)
		xs = self.out_drop(xs)
		if self.skip_connect:
			x = x + xs
		else:
			x = xs
		return x



class SE_Adapter(nn.Module):
	def __init__(self, D_features, mlp_ratio=0.25, out_dim=0, act_layer=nn.GELU, skip_connect=True, drop_path: float=0.0):
		super().__init__()
		self.skip_connect = skip_connect
		D_hidden_features = int(D_features * mlp_ratio)
		self.act = act_layer()
		self.D_fc1 = nn.Linear(D_features, D_hidden_features)
		if out_dim == 0:
			self.D_fc2 = nn.Linear(D_hidden_features, D_features)
		else:
			self.D_fc2 = nn.Linear(D_hidden_features, out_dim)
		print("Adapter reduce rate {}".format(mlp_ratio))
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

	def forward(self, x):
		# x is (B, N, D)
		# Mean-Pooling
		B, N, D = x.shape
		xs = x.mean(dim=1, keepdim=True) # B, 1，D
		xs = self.D_fc1(xs)
		xs = self.act(xs)
		xs = self.D_fc2(xs)
		
		# add sigmoid
		xs = torch.sigmoid(xs)
		xs = xs.expand(-1, N, -1)
		x = x * xs
		#if self.skip_connect:
		#	#print("DEBUG: Adapter Skip Branch")
		#	x = x + self.drop_path(xs)
		#else:
		#	x = self.drop_path(xs)
		return x