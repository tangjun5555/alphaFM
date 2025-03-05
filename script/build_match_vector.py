# -*- coding: utf-8 -*-

import argparse
from typing import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, required=True)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--vector_type", type=str, required=True, choices=["user", "item"])
args = parser.parse_args()

fm_bias = 0.0
fm_w_dict = dict()
fm_v_dict = dict()

with open(args.model_file, mode="r") as fin:
    for line in fin:
        split = line.strip().split()
        name = split[0]
        if name == "bias":
            assert len(split) == 2
            fm_bias = float(split[1])
        else:
            fm_w_dict[name] = float(split[1])
            fm_v_dict[name] = [float(x) for x in split[2:]]
    dim = len(fm_v_dict[name])
    print(f"完成加载模型，FM向量维度:{dim}")


def compute_first_order_score(features: Dict[str, float]):
    res = 0.0
    for k, v in features.items():
        if k not in fm_w_dict:
            continue
        res += fm_w_dict[k] * v
    return res


def compute_second_order_score(features: Dict[str, float]):
    vector_dict = dict()
    for k, v in features.items():
        if k not in fm_v_dict:
            continue
        vector_dict[k] = [x * v for x in fm_v_dict[k]]

    res = 0.0
    if not vector_dict:
        return res
    elif len(vector_dict) == 1:
        return res

    for i in range(dim):
        t1 = 0.0
        t2 = 0.0
        for k, v in vector_dict.items():
            t1 += fm_v_dict[k][i]
            t2 += fm_v_dict[k][i] ** 2
        t1 = t1 ** 2
        res += 0.5 * (t1 - t2)
    return res


def add_vector(features: Dict[str, float]):
    vector_dict = dict()
    for k, v in features.items():
        if k not in fm_v_dict:
            continue
        vector_dict[k] = [x * v for x in fm_v_dict[k]]
    res = [0.0] * dim
    for _, v in vector_dict.items():
        for i in range(dim):
            res[i] += v[i]
    return res


with open(args.input_file, mode="r") as fin:
    with open(args.output_file, mode="w") as fout:
        for line in fin:
            split = line.strip().split("\t")
            v_id = split[0]
            features = {x.split(":")[0]: float(x.split(":")[1]) for x in split[1].strip().split()}

            if args.vector_type == "user":
                fout.write(
                    v_id
                    + "\t"
                    + ",".join([str(fm_bias + compute_first_order_score(features) + compute_second_order_score(features)), str(1.0)] + [str(x) for x in add_vector(features)])
                    + "\n"
                )
            else:
                fout.write(
                    v_id
                    + "\t"
                    + ",".join([str(1.0), str(compute_first_order_score(features) + compute_second_order_score(features))] + [str(x) for x in add_vector(features)])
                    + "\n"
                )
