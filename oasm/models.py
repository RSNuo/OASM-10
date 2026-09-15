"""Context-query multimodal model and controlled structural comparisons.

Feature tokenization and residual blocks follow the previous implementation.
Attention requests no unused weights so PyTorch can use its efficient SDPA path.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .features import modality_indices

class MLPFeatureTokenizer(nn.Module):
    def __init__(self,num_features,d_model):
        super().__init__()
        self.feature_scales=nn.Parameter(torch.ones(num_features)*0.5)
        self.num_projs=nn.ModuleList([nn.Sequential(nn.Linear(1,d_model//2),nn.GELU(),nn.Linear(d_model//2,d_model),nn.LayerNorm(d_model)) for _ in range(num_features)])
        self.pos_embedding=nn.Parameter(torch.randn(1,num_features,d_model)*0.02)
        self.cls_token=nn.Parameter(torch.randn(1,1,d_model)*0.02)
    def forward(self,x):
        scaled=x*F.softplus(self.feature_scales).unsqueeze(0)
        z=torch.stack([proj(scaled[:,i:i+1]) for i,proj in enumerate(self.num_projs)],dim=1)+self.pos_embedding
        return torch.cat([self.cls_token.expand(x.shape[0],-1,-1),z],dim=1)

class EncoderLayer(nn.Module):
    def __init__(self,d,h,ff,drop):
        super().__init__();self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.drop1=nn.Dropout(drop)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d),nn.Dropout(drop))
    def forward(self,x):
        h=self.norm1(x);x=x+self.drop1(self.attn(h,h,h,need_weights=False)[0])
        return x+self.ffn(self.norm2(x))

class Encoder(nn.Module):
    def __init__(self,d,h,ff,drop,layers):
        super().__init__();self.layers=nn.ModuleList([EncoderLayer(d,h,ff,drop) for _ in range(layers)])
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        return x

class Fusion(nn.Module):
    def __init__(self,d,h,drop):
        super().__init__();self.cross_attn=nn.MultiheadAttention(d,h,dropout=drop,batch_first=True)
        self.norm=nn.LayerNorm(d);self.drop=nn.Dropout(drop)
        self.ffn=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(drop),nn.Linear(2*d,d))
    def forward(self,q,kv):
        z=q+self.drop(self.cross_attn(q,kv,kv,need_weights=False)[0])
        return (z+self.ffn(self.norm(z))).squeeze(1)

def head(d,drop):
    return nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d//2),nn.GELU(),nn.Dropout(drop),nn.Linear(d//2,d//4),nn.GELU(),nn.Dropout(drop/2),nn.Linear(d//4,1))

class SoilTransformer(nn.Module):
    def __init__(self,numeric,categorical,cat_dims,variant='ma',d_model=168,nhead=8,layers=2,ff=512,dropout=0.25,static_dropout=0.45):
        super().__init__();self.variant=variant;self.groups=modality_indices(numeric)
        self.static_idx=self.groups['static'];self.dynamic_idx=sorted(i for k,v in self.groups.items() if k!='static' for i in v)
        d=d_model
        if variant=='ft':
            self.tokenizer=MLPFeatureTokenizer(len(numeric),d)
            self.cat_tokens=nn.ModuleList([nn.Embedding(k,d,padding_idx=0) for k in cat_dims])
            self.encoder=Encoder(d,nhead,ff,dropout,layers);self.head=head(d,dropout)
            return
        if variant!='no_static':
            dims=[int(min(16,round(np.sqrt(k))+1)) for k in cat_dims]
            self.embeddings=nn.ModuleList([nn.Embedding(k,e,padding_idx=0) for k,e in zip(cat_dims,dims)])
            self.static_dropout=nn.Dropout(static_dropout)
            self.static_proj=nn.Sequential(nn.Linear(len(self.static_idx)+sum(dims),d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(dropout))
        if variant!='static_only':
            self.encoder=Encoder(d,nhead,ff,dropout,layers)
            if variant=='no_modality':self.tokenizer=MLPFeatureTokenizer(len(self.dynamic_idx),d)
            else:
                self.tokenizers=nn.ModuleDict({k:MLPFeatureTokenizer(len(v),d) for k,v in self.groups.items() if k!='static' and v})
                self.modality_emb=nn.ParameterDict({k:nn.Parameter(torch.randn(1,1,d)*.02) for k in self.tokenizers})
            if variant=='concat':self.concat_fuse=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU())
            else:self.fusion=Fusion(d,nhead,dropout)
            if variant=='no_static':self.query=nn.Parameter(torch.randn(1,1,d)*.02)
        self.head=head(d,dropout)
    def forward(self,x,c):
        if self.variant=='ft':
            z=torch.cat([self.tokenizer(x)]+[e(c[:,i]).unsqueeze(1) for i,e in enumerate(self.cat_tokens)],dim=1)
            return self.head(self.encoder(z)[:,0]).squeeze(1)
        if self.variant!='no_static':
            z=torch.cat([x[:,self.static_idx]]+[e(c[:,i]) for i,e in enumerate(self.embeddings)],dim=1)
            s=self.static_proj(self.static_dropout(z))
        if self.variant=='static_only':return self.head(s).squeeze(1)
        if self.variant=='no_modality':tokens=self.tokenizer(x[:,self.dynamic_idx])
        else:tokens=torch.cat([t(x[:,self.groups[k]])+self.modality_emb[k] for k,t in self.tokenizers.items()],dim=1)
        z=self.encoder(tokens)
        if self.variant=='concat':h=self.concat_fuse(torch.cat([s,z.mean(dim=1)],dim=1))
        else:
            q=self.query.expand(x.shape[0],-1,-1) if self.variant=='no_static' else s.unsqueeze(1)
            h=self.fusion(q,z)
        return self.head(h).squeeze(1)

def model_config(depth):
    drop,static={5:(.25,.45),20:(.30,.50),50:(.35,.55)}[int(depth)]
    return dict(d_model=168,nhead=8,layers=2,ff=512,dropout=drop,static_dropout=static)
