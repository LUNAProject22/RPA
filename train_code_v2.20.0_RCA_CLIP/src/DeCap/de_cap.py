import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pickle
from typing import Tuple, Optional, Union
from open_clip import Adapter

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 512):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./DeCap/decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out
    

class DeCap_GPT2(nn.Module):

    def __init__(self,prefix_size: int = 768, prefix_len: int = 1):
        super(DeCap_GPT2, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        self.prefix_len = prefix_len
        self.decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        #with open('./DeCap/decoder_config.pkl','rb') as f:
        #    config = pickle.load(f)
        #self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, self.embedding_size))
        #self.clip_project = MLP((prefix_size, (self.embedding_size * self.prefix_len) // 3, self.embedding_size * self.prefix_len ))
        #self.clip_project = MLP((prefix_size, self.embedding_size * self.prefix_len ))
        self.drop_out = nn.Dropout(p=0.2)        


    def forward(self, clip_features,gpt_tokens, mask=None, labels=None):
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = self.drop_out(embedding_clip) #  Add dropout for CLIP embedding
        embedding_clip = embedding_clip.reshape(-1, self.prefix_len, self.embedding_size)

        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat, attention_mask=mask, labels=labels)
        return out
    
#    def train(self, mode: bool = True):
#        super(DeCap_GPT2, self).train(mode)
#        self.decoder.eval()
#        return self
#    
#    def parameters(self, recurse: bool = True):
#        return self.clip_project.parameters()


class Adaptered(nn.Module):
    def __init__(self, orig_layer, D_features=768, mlp_ratio=0.5, skip_connect=True):
        super().__init__()
        self.orig_layer = orig_layer
        self.Adapter = Adapter(D_features=D_features, mlp_ratio=mlp_ratio, skip_connect=skip_connect, drop_path=0.2)
        self.init_parameters()

    def init_parameters(self):
        for n2, m2 in self.Adapter.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Linear):
                    print("Init GPT-2 Adatper D_fc2 with zeros")
                    nn.init.constant_(m2.weight, 0)
                    nn.init.constant_(m2.bias, 0)

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        #print(orig_out.shape)
        output = self.Adapter.forward(orig_out)

        return output


class DeCap_GPT2_Adapter(nn.Module):

    def __init__(self,prefix_size: int = 768, prefix_len: int = 1):
        super(DeCap_GPT2_Adapter, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        self.prefix_len = prefix_len
        self.decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        #self.clip_project = MLP((prefix_size, self.embedding_size * self.prefix_len ))
        self.clip_project = MLP((prefix_size,self.embedding_size))
        #self.clip_project = MLP((prefix_size, (self.embedding_size * self.prefix_len) // 3, self.embedding_size * self.prefix_len ))

        # Insert Adapter
        print(self.decoder)
        for i in range(12):
            self.decoder.transformer.h[i].attn.c_proj = Adaptered(self.decoder.transformer.h[i].attn.c_proj)
            self.decoder.transformer.h[i].mlp.c_proj = Adaptered(self.decoder.transformer.h[i].mlp.c_proj)        


    def forward(self, clip_features,gpt_tokens, mask=None, labels=None):
        #print(clip_features.shape)
        embedding_clip = self.clip_project(clip_features)
        #embedding_clip = clip_features
        embedding_clip = embedding_clip.reshape(-1, self.prefix_len, self.embedding_size)

        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat, attention_mask=mask, labels=labels)
        return out
    

