"""Monte Carlo check of ECHOME's adaptive assessment, using the repo's own CAT engine.

Run from the root of github.com/Lourdhu02/echome:  python echome_cat_sim.py 500
"""
import json, sys, numpy as np
sys.path.insert(0, '.')
from src.cat_engine import prob_grm, estimate_theta_map, calculate_se, select_next_item
from src.models import QuestionItem, TraitState
bank = [QuestionItem(**d) for d in json.load(open('data/item_bank.json'))]
dims = sorted({i.dimension for i in bank})
rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
used, true_t, cat_t, full_t, ses = [], [], [], [], []
def draw(theta, it):
    p = np.array([prob_grm(theta, it.a, it.b, k) for k in range(len(it.b)+1)]); p = np.clip(p, 0, None)
    return int(rng.choice(len(p), p=p/p.sum()))
for n in range(N):
    for dim in dims:
        items = [i for i in bank if i.dimension == dim]; idx = {i.item_id: i for i in items}
        th = rng.standard_normal()
        resp_all = {i.item_id: draw(th, i) for i in items}
        state = TraitState(dimension=dim, theta=0.0, se=1.0, responses={}, answered_items=[]) if 'dimension' in TraitState.model_fields else TraitState()
        resp = {}
        while len(resp) < 5 and state.se > 0.32:
            it = select_next_item(state, items)
            if it is None: break
            resp[it.item_id] = resp_all[it.item_id]
            state.answered_items.append(it.item_id)
            state.theta = estimate_theta_map(resp, idx); state.se = calculate_se(state.theta, resp, idx)
        used.append(len(resp)); true_t.append(th); cat_t.append(state.theta); ses.append(state.se)
        full_t.append(estimate_theta_map(resp_all, idx))
used=np.array(used); t=np.array(true_t); c=np.array(cat_t); f=np.array(full_t)
print(f"simulees={N} x {len(dims)} traits; items/trait CAT mean={used.mean():.2f} of 10 -> reduction {100*(1-used.mean()/10):.1f}%")
print(f"r(CAT,true)={np.corrcoef(c,t)[0,1]:.3f}  r(full,true)={np.corrcoef(f,t)[0,1]:.3f}  r(CAT,full)={np.corrcoef(c,f)[0,1]:.3f}")
print(f"RMSE CAT={np.sqrt(((c-t)**2).mean()):.3f} full={np.sqrt(((f-t)**2).mean()):.3f}; SE<=0.32 reached {100*(np.array(ses)<=0.32).mean():.1f}%")
