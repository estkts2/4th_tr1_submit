import os

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
    

if __name__ == '__main__':
    
    config_file     = '/home/kts2/gc2020/weights/v4_fixed_f010203041042060708_lr0.00075_10of30_warm2000_resume_epoch_7_/test_config.py'
    checkpoint_file = '/home/kts2/gc2020/weights/v4_fixed_f010203041042060708_lr0.00075_10of30_warm2000_resume_epoch_7_/epoch_27.pth'
    out_pickle_path = './launch_test.pickle'
    out_jon_prefix = './launch_test'
    
    run_test(config_file, checkpoint_file, 
            out_pickle_path, out_jon_prefix)
    