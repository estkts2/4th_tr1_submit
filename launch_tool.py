import os
import multiprocessing as mp

def run_test(config_file, checkpoint_file,
             out_pickle_path, gpu_order=0):
    
    cmd = f"""CUDA_VISIBLE_DEVICES={gpu_order} python \
    /root/VarifocalNet/tools/test.py \
    --out {out_pickle_path} \
    --format-only \
    {config_file} {checkpoint_file} 
    """
    
    print(cmd)
    os.system(cmd)
    
def foo(param):
    run_test(**param)
    
def run_test_mulit(params):
    pool = mp.Pool(processes=2)
    pool.map(foo, params)
    pool.join()
    pool.close()
    