import sys
import gc
import time

for seed in [1,2,3,4,5]:
    script_descriptor = open("train_ensemble.py", encoding ='utf-8')
    a_script = script_descriptor.read()
    sys.argv = ["train_ensemble.py",'--seed',f'{seed}','--experiment','ensemble']
    try:
        print(sys.argv)
        print('start')
        exec(a_script)
        gc.collect()
        time.sleep(5)
    except:
        print('failed')

for seed in [1,2,3,4,5]:
    script_descriptor = open("train_ensemble_AT.py", encoding ='utf-8')
    a_script = script_descriptor.read()
    sys.argv = ["train_ensemble_AT.py",'--seed',f'{seed}','--experiment','ensemble+AT']
    try:
        print(sys.argv)
        print('start')
        exec(a_script)
        gc.collect()
        time.sleep(5)
    except:
        print('failed')

