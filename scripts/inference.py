# -*- coding: utf-8 -*-

import math
from typing import Dict


class FMRankModel(object):
    def __init__(self, model_file: str):
        self.fm_bias = 0.0
        self.fm_w_dict = dict()
        self.fm_v_dict = dict()
        self.vector_dim = 0

        self.model_file = model_file
        self._load_model()

    def _load_model(self):
        with open(self.model_file, mode="r") as fin:
            for line in fin:
                split = line.strip().split()
                name = split[0]
                if name == "bias":
                    assert len(split) == 2
                    self.fm_bias = float(split[1])
                else:
                    self.fm_w_dict[name] = float(split[1])
                    self.fm_v_dict[name] = [float(x) for x in split[2:]]
            self.vector_dim = len(self.fm_v_dict[name])

    def predict(self, features: Dict[str, float]):
        res = self.fm_bias
        vector_dict = dict()
        for k, v in features.items():
            if k not in self.fm_w_dict:
                continue
            res += self.fm_w_dict[k] * v
            vector_dict[k] = [x * v for x in self.fm_v_dict[k]]
        if len(vector_dict) <= 1:
            return res
        for i in range(self.vector_dim):
            t1 = 0.0
            t2 = 0.0
            for _, v in vector_dict.items():
                t1 += v[i]
                t2 += v[i] ** 2
            t1 = t1 ** 2
            res += 0.5 * (t1 - t2)
        return res

    def predict_prob(self, features: Dict[str, float]):
        res = self.predict(features)
        res = 1.0 / (1.0 + math.exp(-res))
        return res


class FMRecallModel(FMRankModel):
    def __init__(self, model_file: str):
        super().__init__(model_file)

    def _add_vector(self, features: Dict[str, float]):
        vector_dict = dict()
        for k, v in features.items():
            if k not in self.fm_v_dict:
                continue
            vector_dict[k] = [x * v for x in self.fm_v_dict[k]]
        res = [0.0] * self.vector_dim
        for _, v in vector_dict.items():
            for i in range(self.vector_dim):
                res[i] += v[i]
        return res

    def build_user_vector(self, user_features: Dict[str, float]):
        return [self.predict(user_features), 1.0] + self._add_vector(user_features)

    def build_item_vector(self, item_features: Dict[str, float]):
        return [1.0, self.predict(item_features) - self.fm_bias] + self._add_vector(item_features)
