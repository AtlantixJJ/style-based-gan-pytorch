import os

dirs = ["celebahq1", "ffhq1", "bce", "l1", "l1_bce", "l1_pos", "l1_pos_bce", "l1_bce_stddev"]

for d in dirs:
    print(d)
    os.system(f"python script/summarize_evaluation.py record/{d}")