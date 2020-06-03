import os
gpu = 5
cmds = [
    f"python figure/sample_face.py --gpu {gpu} --name face_fewshot_part --n 1,4 --m 7,8,9 --r 3",
    f"python figure/sample_face.py --gpu {gpu} --name face_fewshot_full_1 --n 1,2 --m 0,1,2,3,4,5 --r 4",
    f"python figure/sample_face.py --gpu {gpu} --name face_fewshot_full_2 --n 4,8 --m 0,1,2,3,4,5 --r 4"]
for cmd in cmds:
    os.system(cmd)