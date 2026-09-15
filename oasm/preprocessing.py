"""Fitted objects see explicitly permitted training records only."""
import numpy as np
from sklearn.preprocessing import StandardScaler,PowerTransformer,RobustScaler,QuantileTransformer
from .features import BASE,modality_indices,normalize_category

class Preprocessor:
    def __init__(self,numeric,categorical):
        self.numeric=list(numeric);self.categorical=list(categorical)
    def fit(self,df):
        self.fitted_site_ids=sorted(df.physical_site_id.unique())
        x=df[self.numeric].to_numpy(dtype=float)
        if np.isinf(x).any():raise ValueError('Infinite raw predictors')
        self.fill=np.nanmean(x,axis=0);self.fill=np.nan_to_num(self.fill,nan=0.0)
        x=np.where(np.isnan(x),self.fill,x)
        self.base_idx=[i for i,c in enumerate(self.numeric) if c in BASE]
        self.prescaler=StandardScaler().fit(x[:,self.base_idx]) if self.base_idx else None
        if self.prescaler is not None:x[:,self.base_idx]=self.prescaler.transform(x[:,self.base_idx])
        self.groups=modality_indices(self.numeric);self.scalers={};self.active={}
        for name,idx in self.groups.items():
            if not idx:continue
            active=[i for i in idx if np.std(x[:,i])>1e-10]
            self.active[name]=active
            if not active:continue
            if name in ['sar','optical_thermal','surface_sm']:s=PowerTransformer(method='yeo-johnson',standardize=True)
            elif name=='temporal':s=RobustScaler()
            else:s=QuantileTransformer(n_quantiles=min(1000,len(x)),output_distribution='normal',random_state=42)
            s.fit(x[:,active]);self.scalers[name]=s
        self.vocabulary={}
        for c in self.categorical:
            values=sorted(set(df[c].map(normalize_category)))
            self.vocabulary[c]={v:i+1 for i,v in enumerate(values)}
        self.cat_dims=[len(self.vocabulary[c])+1 for c in self.categorical]
        return self
    def transform(self,df):
        x=df[self.numeric].to_numpy(dtype=float)
        if np.isinf(x).any():raise ValueError('Infinite raw predictors')
        x=np.where(np.isnan(x),self.fill,x)
        if self.prescaler is not None:x[:,self.base_idx]=self.prescaler.transform(x[:,self.base_idx])
        for name,idx in self.groups.items():
            active=self.active.get(name,[])
            if active:x[:,active]=self.scalers[name].transform(x[:,active])
            constant=list(set(idx)-set(active))
            if constant:x[:,constant]=0
        cats=np.column_stack([df[c].map(normalize_category).map(self.vocabulary[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.categorical]) if self.categorical else np.zeros((len(df),0),dtype=np.int64)
        if not np.isfinite(x).all():raise ValueError('Non-finite transformed predictors')
        return x.astype(np.float32),cats

def assert_excluded(prep,heldout):
    overlap=set(prep.fitted_site_ids)&set(heldout)
    if overlap:raise AssertionError(f'Preprocessing includes evaluation sites: {sorted(overlap)[:5]}')
