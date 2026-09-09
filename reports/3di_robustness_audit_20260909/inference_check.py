import sys,json,random,hashlib
from pathlib import Path
import torch
from csubst import structural_prediction as sp
root=Path.cwd();torch.set_num_threads(4);rng=random.Random(927)
seqs=['A','X','XX','M'*3,'X'*31]+[aa*37 for aa in 'ACDEFGHIKLMNPQRSTVWY']+[''.join(rng.choices('ACDEFGHIKLMNPQRSTVWYX',k=n)) for n in [2,3,7,15,31,63,127,255,511,1025]]
results={}
for backend in ['esm3di-35m','prostt5-cnn']:
 p=sp.load_encoder_predictor({'sa_backend':backend,'prostt5_device':'cpu','prostt5_no_download':True})
 with torch.inference_mode():
  single=[p.predict_batch([s])[0] for s in seqs]
  mixed=[]
  for i in range(0,len(seqs),4):mixed.extend(p.predict_batch(seqs[i:i+4]))
  assert all(len(s)==len(o) for s,o in zip(seqs,single))
  result={'sequences':len(seqs),'residues':sum(map(len,seqs)),'batch_singleton_differences':sum(sum(a!=b for a,b in zip(x,y)) for x,y in zip(single,mixed)),'finite_logits':True,'lengths_correct':True}
  assert result['batch_singleton_differences']==0,result
 results[backend]=result
 print(backend,result,flush=True)
 del p
(root/'inference.json').write_text(json.dumps(results,indent=2)+'\n')
