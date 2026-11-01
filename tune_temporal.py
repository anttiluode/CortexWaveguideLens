"""Preregistered-stage-2 validation only: choose hyperparameters before testing."""
import json
from pathlib import Path
from temporal_wavecopy import evaluate

val_seeds=range(100,112)
choices=[]
for q in (0.0, .001, .005, .02):
    for reset in (1.3, 1.7, 2.2):
        _,summ,info=evaluate(val_seeds,q=q,reset_threshold=reset)
        choices.append({'q':q,'reset_threshold':reset,'mse':summ['phase_copy_inferred_motion']['all']['mse'],'resets':info['mean_resets']})
choice=min(choices,key=lambda x:x['mse'])
fixed=[]
for a in (.15,.30,.5):
    _,s,_=evaluate(val_seeds,alpha=a)
    fixed.append({'alpha':a,'mse':s['motion_ema_same_reset']['all']['mse']})
best_a=min(fixed,key=lambda x:x['mse'])
blurs=[]
for b in (.6,1.0,1.5):
    _,s,_=evaluate(val_seeds,blur=b)
    blurs.append({'blur':b,'mse':s['spatial_blur']['all']['mse']})
best_b=min(blurs,key=lambda x:x['mse'])
result={'validation_seeds':[100,111], 'all_choices':choices,'chosen_phase':choice,'chosen_ema':best_a,'chosen_spatial':best_b}
Path(__file__).with_name('results').joinpath('temporal_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('validation selection',choice,best_a,best_b)
