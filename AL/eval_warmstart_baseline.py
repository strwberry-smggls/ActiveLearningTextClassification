import sys
import os
import numpy as np

eval_path = sys.argv[1]
eval_mode = "_".join([eval_path.split("/")[-1], eval_path.split("/")[-2]])

paths = [
    f"{eval_path}_run{x}" for x in range(1, 4) if os.path.isdir(f"{eval_path}_run{x}")
]

result_dict_macro = {}
result_dict_micro = {}

for p in paths:
    output_files = [f for f in os.listdir(p) if f.endswith("_results.txt")]
    for o in output_files:
        with open(os.path.join(p, o), "r") as f:
            f_content = f.read().split("\n")
            micro = f_content[0].split("=")[-1]
            macro = f_content[1].split("=")[-1]
            sent_number = int(o.strip("_results.txt"))
            try:
                result_dict_macro[sent_number].append(float(macro))
            except:
                result_dict_macro[sent_number] = [float(macro)]

            try:
                result_dict_micro[sent_number].append(float(micro))
            except:
                result_dict_micro[sent_number] = [float(micro)]

macro_out = ""
for k in sorted(result_dict_macro.keys()):
    avg_macro = np.mean(result_dict_macro[k])
    macro_out += f"{k},{avg_macro}\n"

with open(f"eval/{eval_mode}_macro.txt", "w") as f:
    f.write(macro_out)


micro_out = ""
for k in sorted(result_dict_micro.keys()):
    avg_micro = np.mean(result_dict_micro[k])
    micro_out += f"{k},{avg_micro}\n"

with open(f"eval/{eval_mode}_micro.txt", "w") as f:
    f.write(micro_out)
