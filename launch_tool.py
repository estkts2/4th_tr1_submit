import os

def get_free_port_():
    try:
        import socket
        from contextlib import closing
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    except:
        return 45312
    
def run_test(config_file, checkpoint_file,
             out_pickle_path, num_gpu=1):
    
    port = get_free_port_()
    
    cmd = f"""python -m torch.distributed.launch --nproc_per_node={num_gpu} --master_port={port}\
    /root/VarifocalNet/tools/test.py \
    --out {out_pickle_path} \
    --format-only \
    --launcher pytorch \
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
    