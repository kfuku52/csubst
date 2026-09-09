import sys, json, random
from pathlib import Path
import torch
from csubst import structural_prediction as sp
root=Path.cwd();torch.set_num_threads(4);rng=random.Random(123)
seqs=['X'*37,'A'*37]+[''.join(rng.choices('ACDEFGHIKLMNPQRSTVWYX',k=n)) for n in [1,2,3,7,31,128,256,512,1025]]
out={}
for device in ['cpu','mps']:
 p=sp.load_encoder_predictor({'prostt5_device':device,'prostt5_no_download':True})
 with torch.inference_mode():
  out[device]=sum([p.predict_batch(seqs[i:i+4]) for i in range(0,len(seqs),4)],[])
 del p
assert out['cpu']==out['mps']
(root/'device.json').write_text(json.dumps({'sequences':len(seqs),'residues':sum(map(len,seqs)),'cpu_mps_predictions_identical':True},indent=2)+'\n')
print('CPU/MPS mixed unknown, low-complexity, random inputs: PASS')
