"""Reproduce model training, physics checks, or the narrated-by-captions demo."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
from perception import ColorAI
from simulation import Cell,train_ai

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['train','check','preview','video'])
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--output',type=Path,default=ROOT/'demo.mp4')
    args=ap.parse_args()
    (ROOT/'evidence').mkdir(exist_ok=True)
    if args.mode=='train':
        _,metrics=train_ai(ROOT); print(json.dumps(metrics,indent=2)); return
    ai=ColorAI.load(ROOT/'models/color_ai.npz')
    if args.mode=='video':
        from video import Movie
        movie=Movie(args.output,ROOT)
        cell=Cell(ai,args.seed,movie.frame)
        movie.intro(cell)
        result=cell.run()
        movie.outro(cell,result); movie.close()
    else:
        cell=Cell(ai,args.seed)
        if args.mode=='preview':
            ds=cell.sense(); print(json.dumps(ds,indent=2))
            Image.fromarray(cell.render()).save(ROOT/'evidence/scene.png')
            Image.fromarray(cell.latest_camera).save(ROOT/'evidence/camera.png')
            cell.close(); return
        result=cell.run()
    (ROOT/f'evidence/run_seed_{args.seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    Image.fromarray(cell.render()).save(ROOT/f'evidence/final_seed_{args.seed}.png')
    cell.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','trajectory']},indent=2))
    assert result['successful']==result['objects']==6, 'Sorting check failed; inspect run evidence.'

if __name__=='__main__': main()
