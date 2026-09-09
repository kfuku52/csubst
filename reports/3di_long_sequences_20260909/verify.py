import argparse,sys,json,time,resource,hashlib,gc
from pathlib import Path
import torch,transformers,peft
from csubst import structural_prediction as sp
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--max-length',type=int,default=8192);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)
root=a.output_dir
root.mkdir(parents=True,exist_ok=True)
source=''.join((Path(__file__).parent/'dystrophin.fasta').read_text().splitlines()[1:])
assert len(source)==3685
predictor=sp.load_encoder_predictor({'prostt5_device':a.device,'prostt5_no_download':True})
result={'device':str(predictor.device),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'threads':torch.get_num_threads(),'position_embedding_type':predictor.model.config.position_embedding_type,'max_position_embeddings':predictor.model.config.max_position_embeddings,'attention_implementation':predictor.model.config._attn_implementation,'source_url':'https://rest.uniprot.org/uniprotkb/P11532.fasta','source_length':len(source),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':[]}
print({k:v for k,v in result.items() if k!='cases'},flush=True)
lengths=[1022,1023,1024,1025,2048,3685,4096,8192]
lengths=[n for n in lengths if n<=a.max_length]
short=source[:128]
with torch.inference_mode():
 control=predictor.predict_batch([short])[0]
 for length in lengths:
  seq=(source*((length+len(source)-1)//len(source)))[:length]
  token_count=int(predictor.tokenizer(seq,return_tensors='pt',truncation=False)['attention_mask'].sum())
  case={'residues':length,'tokens':token_count,'input_kind':'complete natural sequence' if length==len(source) else ('natural sequence prefix' if length<len(source) else 'repeated natural sequence'), 'input_sha256':hashlib.sha256(seq.encode()).hexdigest()}
  start=time.perf_counter()
  try:
   predicted=predictor.predict_batch([seq])[0]
   if a.device=='mps':torch.mps.synchronize()
   assert len(predicted)==length and set(predicted)<=set('ACDEFGHIKLMNPQRSTVWY')
   case.update(success=True,output_length=len(predicted),finite_logits=True,output_sha256=hashlib.sha256(predicted.encode()).hexdigest())
   (root/(a.device+'-'+str(length)+'.fa')).write_text('>'+str(length)+'\n'+predicted+'\n')
  except Exception as exc:
   case.update(success=False,error_type=type(exc).__name__,error=str(exc))
  case['seconds']=time.perf_counter()-start
  case['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  result['cases'].append(case)
  (root/(a.device+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  print(case,flush=True)
  gc.collect()
  if a.device=='mps':torch.mps.empty_cache()
 result['short_prediction_unchanged_after_long_inputs']=predictor.predict_batch([short])[0]==control
 assert result['short_prediction_unchanged_after_long_inputs']
(root/(a.device+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print('Complete',flush=True)
