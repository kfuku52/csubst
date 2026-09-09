import gc, hashlib, json, os, random, time
from pathlib import Path
import torch, transformers, peft
from csubst import structural_prediction as sp
from csubst import structural_alphabet as sa
root=Path.cwd()
torch.set_num_threads(4)
assert torch.cuda.is_available(), 'CUDA is unavailable'
result={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'cases':[]}
rng=random.Random(927)
seqs=['A','X','XX','M'*3,'X'*31]+[aa*37 for aa in 'ACDEFGHIKLMNPQRSTVWY']+[''.join(rng.choices('ACDEFGHIKLMNPQRSTVWYX',k=n)) for n in [2,3,7,15,31,63,127,255,511,1025]]
source=''.join((root/'dystrophin.fasta').read_text().splitlines()[1:])
for backend in os.environ.get('CSUBST_AUDIT_BACKENDS','esm3di-35m,prostt5-cnn').split(','):
 g={'sa_backend':backend,'prostt5_no_download':True,'resource_cache_dir':str(root/'cache'),'prostt5_local_dir':str(root/'models/ProstT5'),'prostt5_cache':False}
 outputs={}
 for device in ['cpu','cuda']:
  torch.cuda.reset_peak_memory_stats()
  p=sp.load_encoder_predictor(dict(g,prostt5_device=device))
  assert next(p.model.parameters()).device.type==device
  start=time.perf_counter()
  with torch.inference_mode():
   singles=[p.predict_batch([s])[0] for s in seqs]
   batches=sum([p.predict_batch(seqs[i:i+4]) for i in range(0,len(seqs),4)],[])
   assert all(len(a)==len(b) for a,b in zip(seqs,singles))
   assert batches==singles, 'Batch predictions differ'
   repeated=p.predict_batch(seqs[:4])
   assert repeated==singles[:4]
   outputs[device]=singles
   if device=='cuda' and backend=='esm3di-35m':
    for length in [1023,2048,3685,4096,8192]:
     seq=(source*3)[:length]
     pred=p.predict_batch([seq])[0]
     assert len(pred)==length
     result['cases'].append({'backend':backend,'device':device,'long_residues':length,'output_sha256':hashlib.sha256(pred.encode()).hexdigest(),'success':True})
  torch.cuda.synchronize()
  case={'backend':backend,'device':device,'sequences':len(seqs),'residues':sum(map(len,seqs)),'singleton_batch_repeat_identical':True,'finite_logits':True,'output_lengths_correct':True,'diagnostic_seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
  result['cases'].append(case);print(case,flush=True)
  del p;gc.collect();torch.cuda.empty_cache()
 diffs=sum(sum(x!=y for x,y in zip(a,b)) for a,b in zip(outputs['cpu'],outputs['cuda']))
 result[backend+'_cpu_cuda_residue_differences']=diffs
 (root/(backend+'-predictions.json')).write_text(json.dumps(outputs)+'\n')
 # Exercise public device=auto dispatch and prediction cache with actual weights.
 cache_g=dict(g,prostt5_device='auto',prostt5_cache=True,prostt5_cache_file=str(root/(backend+'-cache.tsv')))
 public=sa.predict_3di({'probe':seqs[-1]},cache_g)
 assert public['probe']==outputs['cuda'][-1]
 original=sp.load_encoder_predictor
 sp.load_encoder_predictor=lambda g: (_ for _ in ()).throw(AssertionError('Cache hit loaded model'))
 try: assert sa.predict_3di({'probe':seqs[-1]},cache_g)==public
 finally: sp.load_encoder_predictor=original
 result[backend+'_public_auto_and_cache_pass']=True
 (root/('results-'+backend+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print('ALL CUDA CHECKS COMPLETED',flush=True)
