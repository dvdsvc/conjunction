"""Inspect or explicitly run selected original scripts with relocated data roots."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

TASKS = {
 'build-extended-l3':'research.temporal.build_ext_years_l3',
 'precompute-v15':'research.temporal.precompute_v15',
 'train-v15':'research.temporal.train_temporal_v15',
 'precompute-v15-fix':'research.temporal.fixes.precompute_v15_fix',
 'train-v15-fix':'research.temporal.fixes.train_temporal_v15_fix',
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task',choices=sorted(TASKS));ap.add_argument('--data-root',type=Path)
    ap.add_argument('--papers-root',type=Path);ap.add_argument('--years',type=int,nargs='+');ap.add_argument('--execute',action='store_true')
    args,extra=ap.parse_known_args()
    if not args.task:
        print(json.dumps({'tasks':TASKS,'default':'dry run; --execute is required to invoke a historical script'},indent=2));return
    if not args.data_root:ap.error('--data-root is required')
    data=args.data_root.resolve();temp=data/'temporal_l3'
    overrides={'DATA_DIR':str(data),'TEMP_DIR':str(temp),'OUT_DIR':str(temp/('precomputed_v15_fix' if args.task.endswith('fix') else 'precomputed_v15')),'CONCEPT_IDS_FILE':str(data/'concept_ids_l3.json'),'OLD_CONCEPT_IDS_FILE':str(data/'concept_ids.json')}
    if args.papers_root:
        papers=args.papers_root.resolve();overrides.update(PAPERS_GEOL=str(papers/'geology_papers.jsonl'),PAPERS_CLIM=str(papers/'climate_papers.jsonl'))
    if not args.execute:
        print(json.dumps({'task':args.task,'module':TASKS[args.task],'path_overrides':overrides,'year_cutoffs_override':args.years,'forwarded_arguments':extra,'will_execute':False},indent=2));return
    if not data.is_dir():ap.error('data root does not exist; run without --execute to inspect the plan')
    if args.task=='build-extended-l3' and not args.papers_root:ap.error('--papers-root is required for graph construction')
    if args.task.startswith('train'):
        import torch
        if not torch.cuda.is_available():ap.error('The original full-data trainer/evaluator requires CUDA; use examples.synthetic_pipeline for CPU')
        if '--eval' in extra:
            folder='model_v15_fix_rolling' if args.task.endswith('fix') else 'model_v15_rolling'
            if not (temp/folder/'best.pt').is_file():ap.error('No checkpoint: refusing to launch a potentially uninitialized evaluation')
    module=importlib.import_module(TASKS[args.task])
    for key,value in overrides.items():
        if hasattr(module,key):setattr(module,key,value)
    if args.years and hasattr(module,'YEAR_CUTOFFS'):module.YEAR_CUTOFFS=args.years
    if args.task=='build-extended-l3':temp.mkdir(parents=True,exist_ok=True)
    old=sys.argv
    try:
        sys.argv=[module.__file__,*(extra[1:] if extra[:1]==['--'] else extra)]
        module.main()
    finally:sys.argv=old

if __name__=='__main__':main()
