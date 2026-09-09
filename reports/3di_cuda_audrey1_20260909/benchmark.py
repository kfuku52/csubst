import gc, hashlib, json, os, platform, statistics, time
from pathlib import Path
import torch, transformers, peft
from csubst import structural_prediction as sp
from csubst import sequence
root=Path.cwd();torch.set_num_threads(4)
assert torch.cuda.is_available()
inputs=sequence.read_fasta(str(root/'input.fa'))
sequences=sorted(inputs.values(),key=len,reverse=True)
assert sorted(map(len,sequences))==[418,960]
result={'host':platform.node(),'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),'batch_size':2,'warmup_runs':1,'measured_runs':3,'input_sha256':hashlib.sha256((root/'input.fa').read_bytes()).hexdigest(),'input_lengths':[len(s) for s in sequences],'cache':False,'model_loading_in_timing':False,'cases':[]}
for backend in ['esm3di-35m','prostt5-cnn']:
 reference=None
 for device in ['cpu','cuda']:
  g={'sa_backend':backend,'prostt5_device':device,'prostt5_no_download':True,'resource_cache_dir':str(root/'cache'),'prostt5_local_dir':str(root/'models/ProstT5'),'prostt5_cache':False}
  start=time.perf_counter();predictor=sp.load_encoder_predictor(g)
  if device=='cuda':torch.cuda.synchronize()
  load_seconds=time.perf_counter()-start
  durations=[]
  with torch.inference_mode():
   warm=predictor.predict_batch(sequences)
   for repeat in range(3):
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    pred=predictor.predict_batch(sequences)
    if device=='cuda':torch.cuda.synchronize()
    durations.append(time.perf_counter()-start)
    assert pred==warm
  if reference is None:reference=warm
  assert warm==reference, 'CPU/CUDA predictions differ'
  case={'backend':backend,'device':device,'load_seconds':load_seconds,'seconds':durations,'median_seconds':statistics.median(durations),'output_sha256':hashlib.sha256('\n'.join(warm).encode()).hexdigest(),'output_lengths':[len(p) for p in warm]}
  result['cases'].append(case)
  (root/'benchmark-results.json').write_text(json.dumps(result,indent=2)+'\n')
  print(case,flush=True)
  del predictor;gc.collect();torch.cuda.empty_cache()
result['all_repeats_and_cpu_cuda_outputs_identical']=True
(root/'benchmark-results.json').write_text(json.dumps(result,indent=2)+'\n')
print('BENCHMARK COMPLETE',flush=True)
