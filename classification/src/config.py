import os
import platform

root_dir = "/home/appleternity/workspace/lab/Crowd/CODA-19"
covid_data_dir = os.path.join(root_dir, "data", "CODA19_v1_20200504", "human_label")

model_dir = os.path.join(root_dir, "classification", "model")
log_dir = os.path.join(root_dir, "classification", "log")
result_dir = os.path.join(root_dir, "classification", "result")
cache_dir = os.path.join(root_dir, "classification", "cache")

for folder in [model_dir, log_dir, result_dir, cache_dir]:
    if not os.path.isdir(folder):
        os.mkdir(folder)

label_mapping = {
    "background": 0,
    "purpose": 1,
    "method": 2, 
    "finding": 3,
    "other": 4,
}
