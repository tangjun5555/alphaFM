# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_model_file",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_model_file",
    type=str,
    required=True,
)
parser.add_argument(
    "--vector_dim",
    type=int,
    required=True,
)

args = parser.parse_args()

with open(args.input_model_file, mode="r") as fin, open(args.output_model_file, mode="w") as fout:
    index = 0
    for line in fin:
        index += 1
        split = line.strip().split()
        if index == 1:
            # bias
            fout.write(" ".join(split[:(1 + 1)]) + "\n")
        else:
            tmp = split[1:(1 + 1 + args.vector_dim)]
            for x in tmp:
                try:
                    float(x)
                except Exception as e:
                    print(f"index:{index}, line:{line} is not correct format.")
                    continue
            tmp = list(set(tmp))
            if len(tmp) == 1 and tmp[0] == "0":
                continue
            fout.write(" ".join(split[:(1 + 1 + args.vector_dim)]) + "\n")
        if index % 10000 == 0:
            print(f"finished {index}.")
