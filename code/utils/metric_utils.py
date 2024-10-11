from copy import deepcopy

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
import zss
from nltk import edit_distance
from zss import Nodeion

import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein


def compute_accuracy(df, true_col, pred_col):
    df = deepcopy(df)
    correct_predictions = df[df[true_col] == df[pred_col]]
    accuracy = len(correct_predictions) / len(df)
    return accuracy


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def normalized_rmse(y_true, y_pred):
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    if denominator == 0:
        # force finite
        if numerator == 0:
            ret = 1.0
        else:
            ret = 0.0
    else:
        ret = sigmoid(numerator/denominator)
    return ret


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def _compute_metric(truths, preds, d_type):
    if len(truths) != len(preds):
        return 0.

    if d_type == "categorical":
        # cast datatypes --
        truths = [str(t) for t in truths]
        preds = [str(p) for p in preds]

        return normalized_levenshtein_score(truths, preds)

    elif d_type == "numerical":
        truths = [float(t) for t in truths]
        preds = [float(p) for p in preds]
        return normalized_rmse(truths, preds)
    else:
        raise ValueError


def compute_metrics_counts(true_df, pred_df):
    """
    Evaluate predictions using Benetech - Making Graphs Accessible metric

    :param true_df: ground truth dataframe
    :type true_df: pd.DataFrame
    :param pred_df: _description_
    :type pred_df: pd.DataFrame
    :return: custom metric
    :rtype: float
    """
    true_df = deepcopy(true_df)
    pred_df = deepcopy(pred_df)

    gt_required_cols = ["id", "data_series", "chart_type"]

    for col in gt_required_cols:
        assert col in true_df.columns, f"{col} must be there in true_df"
    true_df = true_df[gt_required_cols].copy()

    true_df = true_df.rename(
        columns={
            "data_series": "true_data_series",
            "chart_type": "true_chart_type",
        }
    )

    true_df["true_data_series"] = true_df["true_data_series"].apply(lambda x: [elem for elem in x if elem == elem])
    true_df["true_count"] = true_df["true_data_series"].apply(lambda x: len(x))

    pred_df = pred_df[["id", "count", "chart_type"]].copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_df = pred_df.rename(
        columns={
            "count": "pred_count",
            "chart_type": "pred_chart_type",
        }
    )

    df = pd.merge(true_df, pred_df, on="id", how="left")

    chart_type_accuracy = compute_accuracy(df, "true_chart_type", "pred_chart_type")
    count_accuracy = compute_accuracy(df, "true_count", "pred_count")

    return_dict = dict()
    return_dict["lb"] = count_accuracy
    return_dict['chart_type_accuracy'] = chart_type_accuracy
    return_dict["count_accuracy"] = count_accuracy

    return return_dict


def compute_metrics(true_df, pred_df):
    """
    Evaluate predictions using Benetech - Making Graphs Accessible metric

    :param true_df: ground truth dataframe
    :type true_df: pd.DataFrame
    :param pred_df: _description_
    :type pred_df: pd.DataFrame
    :return: custom metric
    :rtype: float
    """
    true_df = deepcopy(true_df)
    pred_df = deepcopy(pred_df)

    gt_required_cols = ["id", "source", "data_series", "chart_type", "data_type"]
    for col in gt_required_cols:
        assert col in true_df.columns, f"{col} must be there in true_df"
    true_df = true_df[gt_required_cols].copy()

    true_df = true_df.rename(
        columns={
            "data_series": "true_data_series",
            "chart_type": "true_chart_type",
        }
    )

    pred_df = pred_df[["id", "data_series", "chart_type"]].copy()
    pred_df = pred_df.reset_index(drop=True)

    pred_df = pred_df.rename(
        columns={
            "data_series": "pred_data_series",
            "chart_type": "pred_chart_type",
        }
    )

    df = pd.merge(true_df, pred_df, on="id", how="left")
    df["pred_data_series"] = df["pred_data_series"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    df["pred_chart_type"] = df["pred_chart_type"].apply(
        lambda x: x if isinstance(x, str) else "NotAChart"
    )

    df = df.reset_index(drop=True)
    mga_lb, scores = _get_score(df)

    return_dict = dict()
    return_dict["lb"] = mga_lb
    return_dict['scores'] = scores

    # chart-wise scores
    chart_options = [
        "horizontal_bar",
        "dot",
        "scatter",
        "line",
        "vertical_bar",
    ]

    for ct in chart_options:
        tmp_df = df[df["true_chart_type"] == ct].copy()
        tmp_df = tmp_df.reset_index(drop=True)
        s, _ = _get_score(tmp_df)
        return_dict[f"{ct}_score"] = s

    return return_dict


def _get_score(df):
    df = deepcopy(df)
    if len(df) == 0:
        return -1, []

    scores = []

    for _, row in df.iterrows():
        if row["pred_chart_type"] != row["true_chart_type"]:
            score = 0.0
        else:
            # check for nan in truths ---
            truths = [t for t in row["true_data_series"] if t == t]
            preds = row["pred_data_series"]

            try:
                score = _compute_metric(truths, preds, row["data_type"])
            except Exception as e:
                print(e)
                score = 0.

        scores.append(score)

    mga_lb = np.mean(scores)

    return mga_lb, score


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node


    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )

