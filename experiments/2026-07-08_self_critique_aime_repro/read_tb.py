from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
p='experiments/2026-07-08_self_critique_aime_repro/outputs/grpo_ckpt/tensorboard_log/selfcritique_aime_inject/qwen25_7b_aime_member_inject_20260710133433'
ea=EventAccumulator(p); ea.Reload()
tags=ea.Tags()['scalars']
print('ALL TAGS:')
for t in tags:
    print('  ', t)
print()
for t in tags:
    if re.search('reward|acc|entropy|score|val|kl|response_len|advantage', t, re.I):
        vals=ea.Scalars(t)
        print('---', t)
        print('   ', [(v.step, round(v.value,4)) for v in vals])
