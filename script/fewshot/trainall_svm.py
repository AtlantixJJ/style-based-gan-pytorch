import os
import subprocess, threading

train_size = [16, 32, 64, 256]
layer_index = [3, 4, 5, 6, 7]
cmds = []
cmd_idx = -1
mutex = threading.Lock()
gpus = [5,5,5,6,6,6]
basecmd = "python script/fewshot/svm_bce.py --layer-index {layer} --train-size {ts}"

def fetch():
    global cmd_idx
    # avoid to fetch lock if no resource
    if cmd_idx >= len(cmds):
        return False, ""

    mutex.acquire()

    # make sure there is resource when lock is acquired
    if cmd_idx >= len(cmds):
        mutex.release()
        return False, ""

    # fetch the resource
    cmd_idx += 1

    mutex.release()

    return True, cmds[cmd_idx]

def worker(gpu):
    print(f"=> Worker on gpu {gpu} started")
    while True:
        flag, cmd = fetch()
        if not flag:
            break
        print(f"=> Worker on gpu {gpu} fetched job")
        cmd += f" --gpu {gpu}"
        print(f"=> {cmd}")
        flag = subprocess.call(cmd.split(" "))
        if flag == 0:
            print(f"=> Worker on gpu {gpu} finish job successfully")
        else:
            print(f"=> Worker on gpu {gpu} job error!")
    print(f"=> Worker on gpu {gpu} finished")

def get_all_commands():
    cmds = []
    for ts in train_size:
        for l in layer_index:
            cmds.append(basecmd.format(layer=l, ts=ts))
    return cmds

cmds = get_all_commands()
threads = []
for i in gpus:
    th = threading.Thread(target=worker, args=[i])
    th.start()
    threads.append(th)
for th in threads:
    th.join()
