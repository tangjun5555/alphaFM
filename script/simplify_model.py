# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vector_dim", type=int, required=True)
parser.add_argument("--input_model_file", type=str, required=True)
parser.add_argument("--output_model_file", type=str, required=True)
args = parser.parse_args()

feature_weight = []
with open(args.input_model_file, mode="r") as fin:
    with open(args.output_model_file, mode="w") as fout:
        index = 1
        for line in fin:
            split = line.strip().split()
            if index == 1:
                # bias
                fout.write(" ".join(split[:(1 + 1)]) + "\n")
            else:
                # 模型参数是否为float
                tmp = split[1:(1 + 1 + args.vector_dim)]
                for x in tmp:
                    try:
                        float(x)
                    except:
                        print("模型简化错误", line)
                        continue

                # 模型参数是否都为0
                tmp = list(set(tmp))
                if len(tmp) == 1 and tmp[0] == "0":
                    continue
                # 写入模型参数
                fout.write(" ".join(split[:(1 + 1 + args.vector_dim)]) + "\n")
            index += 1
print("完成模型简化")
