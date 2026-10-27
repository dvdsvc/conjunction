import numpy as np, torch, torch.nn as nn, json, re
DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MAX_NODES=32; EMBED_DIM=68; SCIBERT_DIM=768; HIDDEN=256; N_LAYERS=4; N_HEADS=8; DIST_BINS=10; DIST_EMBED_DIM=32

class PencilV5(nn.Module):
    def __init__(s, se):
        super().__init__()
        s.register_buffer('scibert_table', se)
        s.scibert_proj=nn.Linear(SCIBERT_DIM,HIDDEN//2); s.struct_proj=nn.Linear(EMBED_DIM,HIDDEN//2)
        s.dist_embed=nn.Embedding((DIST_BINS+2)**2,DIST_EMBED_DIM)
        s.combine=nn.Linear(HIDDEN//2+HIDDEN//2+DIST_EMBED_DIM,HIDDEN)
        s.cand_pos_embed=nn.Embedding(64,HIDDEN)
        s.layers=nn.ModuleList()
        for _ in range(N_LAYERS):
            s.layers.append(nn.ModuleDict({'attn':nn.MultiheadAttention(HIDDEN,N_HEADS,batch_first=True,dropout=0.1),'ln1':nn.LayerNorm(HIDDEN),'ffn':nn.Sequential(nn.Linear(HIDDEN,HIDDEN*4),nn.GELU(),nn.Linear(HIDDEN*4,HIDDEN),nn.Dropout(0.1)),'ln2':nn.LayerNorm(HIDDEN),'mp':nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.GELU())}))
        s.ln_f=nn.LayerNorm(HIDDEN); s.head=nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,1))
    def _build_node_features(s,inp,nids):
        B,G,N,_=inp.shape; fi=inp.reshape(B*G*N,-1); fn=nids.reshape(B*G*N)
        xr=fi.clone(); xr[:,2*MAX_NODES+1]=0; xr[:,2*MAX_NODES+2]=0; se=s.struct_proj(xr)
        nf=fn.clamp(0,s.scibert_table.shape[0]-1); sc=s.scibert_proj(s.scibert_table[nf])
        du=(fi[:,2*MAX_NODES+1]*DIST_BINS).long().clamp(0,DIST_BINS+1); dv=(fi[:,2*MAX_NODES+2]*DIST_BINS).long().clamp(0,DIST_BINS+1)
        de=s.dist_embed(du*(DIST_BINS+2)+dv); x=s.combine(torch.cat([se,sc,de],dim=-1))
        ci=torch.arange(G,device=inp.device).unsqueeze(1).expand(-1,N).reshape(-1).unsqueeze(0).expand(B,-1).reshape(-1)
        x=x+s.cand_pos_embed(ci); return x.reshape(B,G*N,-1)
    def _build_iso_mask(s,nc,nn,dev):
        S=(1+nc)*nn; m=torch.full((S,S),-10000.0,device=dev); se=nn; m[:se,:]=0.0
        for k in range(1+nc):
            st=k*nn; en=(k+1)*nn; m[st:en,:se]=0.0; m[st:en,st:en]=0.0
        return m
    def forward(s,inp,nids):
        B,G,N,_=inp.shape; dev=inp.device; x=s._build_node_features(inp,nids)
        adj=inp[:,:,:,MAX_NODES:MAX_NODES+N]; deg=adj.sum(dim=3,keepdim=True).clamp(min=1); an=adj/deg
        pm=(inp[:,:,:,:2*MAX_NODES+1].abs().sum(dim=3)==0); pf=pm.reshape(B,G*N)
        im=s._build_iso_mask(G-1,N,dev)
        for l in s.layers:
            h=l['ln1'](x); h,_=l['attn'](h,h,h,attn_mask=im,key_padding_mask=pf); x=x+h
            xc=x.reshape(B,G,N,-1); hm=l['mp'](torch.einsum('bgnh,bgnm->bgmh',xc,an)); xc=xc+hm
            x=x.reshape(B,G*N,-1); x[:,:]=xc.reshape(B,G*N,-1); h=l['ln2'](x); x=x+l['ffn'](h)
        x=s.ln_f(x); xc=x.reshape(B,G,N,-1); v=(~pm).unsqueeze(-1).float()
        p=(xc*v).sum(dim=2)/v.sum(dim=2).clamp(min=1); return s.head(p).squeeze(-1)

device=torch.device('cuda')
scibert=torch.load(f'{DATA_DIR}/concept_scibert.pt',weights_only=True)
model=PencilV5(scibert).to(device)
model.load_state_dict(torch.load(f'{DATA_DIR}/pencil_v5_iso_best.pt',weights_only=True))
model.eval()

val_data=np.load(f'{DATA_DIR}/val_precomp_v3.npz')
val_inp=torch.from_numpy(val_data['inp'].astype(np.float32)).reshape(-1,6,32,68)
val_nids=torch.from_numpy(val_data['node_ids']).reshape(-1,6,32)

N=200

# Test 1: Zero input
print("=== Test D: Zero Input ===")
zi=torch.zeros(2,6,32,68,device=device); zn=torch.zeros(2,6,32,dtype=torch.long,device=device)
zi[:,:,0,0]=1.0
with torch.inference_mode(): zs=model(zi,zn).cpu().numpy()
for g in range(6): print(f"  cand[{g}]: {zs[0,g]:.4f}")

# Test 2: Identical subgraphs
print("\n=== Test E: Identical Subgraphs ===")
si=val_inp[:N].clone(); sn=val_nids[:N].clone()
for g in range(1,6): si[:,g]=si[:,0]; sn[:,g]=sn[:,0]
with torch.inference_mode(): ss=model(si.to(device),sn.to(device)).cpu().numpy()
for g in range(6): print(f"  cand[{g}]: {ss[:20,g].mean():.4f}")
print(f"  source - cand1: {(ss[:,0]-ss[:,1]).mean():.4f}")

# Test 3: Swap pos/neg positions
print("\n=== Test B: Swap Pos/Neg Positions ===")
inp_sw=val_inp[:N].clone(); nids_sw=val_nids[:N].clone()
tmp_i=inp_sw[:,0].clone(); tmp_n=nids_sw[:,0].clone()
inp_sw[:,0]=inp_sw[:,1]; inp_sw[:,1]=tmp_i
nids_sw[:,0]=val_nids[:N,1]; nids_sw[:,1]=tmp_n
with torch.inference_mode():
    orig=model(val_inp[:N].to(device),val_nids[:N].to(device)).cpu().numpy()
    sw=model(inp_sw.to(device),nids_sw.to(device)).cpu().numpy()
print(f"  Original: src(=pos)={orig[:,0].mean():.3f} cand(=neg)={orig[:,1].mean():.3f}")
print(f"  Swapped:  src(=neg)={sw[:,0].mean():.3f} cand(=pos)={sw[:,1].mean():.3f}")
print(f"  src > cand after swap: {(sw[:,0]>sw[:,1]).mean():.4f}")

# Test 4: Random noise
print("\n=== Test F: Random Noise ===")
rng=np.random.RandomState(42)
ri=torch.from_numpy(rng.randn(N,6,32,68).astype(np.float32)).to(device)
rn=torch.from_numpy(rng.randint(0,1000,(N,6,32)).astype(np.int64)).to(device)
with torch.inference_mode(): rs=model(ri,rn).cpu().numpy()
for g in range(6): print(f"  cand[{g}]: {rs[:,g].mean():.4f}")

print("\nDone.")
