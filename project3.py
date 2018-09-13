"""
project3.py

Authors: Dhaval Chauhan dmc8686@g.rit.edu
Authors: Bhavin Bhuta bsb5375@g.rit.edu

Preprocess CROHME dataset, extract features and do
classification, segmentation, parsing of symbols
in expressions using random forest.
"""

import xml.etree.ElementTree as ET
from pylab import *
import numpy as np
from copy import deepcopy
from scipy import interpolate
import sys
import os
import shutil
import random
import csv
import _pickle as cPickle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from project1 import *
from project2 import *


FILENAME = "randomforest.sav"
FILENAME_REL = "rforest_relations.sav"
maindatapath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\TrainINKML\TrainINKML"
bonusdatapath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\\bonus_inkml_symbols"
testtestinkmlpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\TestTestINKML"
testtraininkmlpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\TestTrainINKML"
testtestlgpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\TestTestLG"
testtrainlgpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\TestTrainLG"
predictedTestLG_Symbols = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedTestLG_Symbols"
predictedTrainLG_Symbols = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedTrainLG_Symbols"
predictedTestLG_Strokes = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedTestLG_Strokes"
predictedTrainLG_Strokes = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedTrainLG_Strokes"
predictedBonusLG_Symbols = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedBonusLG_Symbols"
predictedBonusLG_Strokes = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\RealProject\.idea\PredictedBonusLG_Strokes"


def producebulkparsedLGfiles(filenames, listofobj_ids, all_expressions, listofstrokeids, listofrelations, lgfilessavedirectory):
    """
    Creates output LG files of the predicted expressions
    :param data_test_filenames_packed: filenames
    :param all_expressions: expresssion predicted
    :param bulk_segmentation_strokeids: list of list of stroke ids
    :param lgfilessavedirectory: output save directory
    :return: None
    """
    i = 0
    if not os.path.exists(lgfilessavedirectory):
        os.makedirs(lgfilessavedirectory)
    for file_name in filenames:
        orig_stdout = sys.stdout
        # output_name = os.path.splitext(file_name)[0]

        # creates a new file using the .inkml filename, with the .lg extension
        filename = getjustfilenamewithoutpathorextension(file_name)
        f = open(lgfilessavedirectory + "\\" + filename + ".lg", "w")
        sys.stdout = f

        # if len(all_expressions[i]) != len(listofrelations[i]) + 1:
        #     print("# NOT A TREE")

        # Write Objects (Segmentation)
        for j in range(len(all_expressions[i])):
            sym = str(all_expressions[i][j])
            o_id = str(listofobj_ids[i][j])
            if sym == ',':
                sym = "COMMA"
            o_id = o_id.replace(',',"COMMA")
            print("O, " + o_id + ", " + str(sym) + ", " + str(1.0) + ", " + str(
                listofstrokeids[i][j]).strip('[]'))
        print()
        # Write Relations (Parsing)
        for j in range(len(listofrelations[i])):
            print(listofrelations[i][j])

        i = i + 1
        sys.stdout = orig_stdout
    f.close()
    return None


def getpredictedrelationship(model, fea):
    """
    Given a feature vector, returns top then predictions
    :param model: classifier model
    :param fea: feature vector
    :return: top ten predictions
    """
    rels_predicted = gettoptenoutputsfromrforest(model, fea)
    return rels_predicted


def getlargerdimensionrange(minx, miny, maxx, maxy):
    """
    Given a bounding box, returns the larger dimension
    from width and height
    :param minx: left edge
    :param miny: top edge
    :param maxx: right edge
    :param maxy: bottom edge
    :return: larger dimension
    """
    w = maxx - minx
    h = maxy - miny
    if  w > h:
        return w, 'w'
    return h, 'h'


def getleftmostrootsymbol(traces, obj_ids, criteria):
    """
    given a list of symbol traces, returns the leftmost
    symbol and the objectid associated with it based on
    Bounding Box or Normalization Square Box criteria
    :param traces: list of traces
    :param obj_ids: list of object ids
    :param criteria: Bounding Box or Normalization Square Box
    :return: leftm0st symbol ids
    """
    leftmost_sym_id = -1
    leftmost_sym_obid = None
    leftmost_sym_val = 9999999
    for ooi in range(len(traces)):
        trc = traces[ooi]
        minx, miny, maxx, maxy = getminmaxofXandY(trc)
        if criteria == "SquareBox":
            larger_dim, H_W = getlargerdimensionrange(minx, miny, maxx, maxy)
            if H_W == 'h':
                minx -= larger_dim/2
                maxx += larger_dim/2
        if minx < leftmost_sym_val:
            leftmost_sym_id = ooi
            leftmost_sym_obid = obj_ids[ooi]
            leftmost_sym_val = minx

    return leftmost_sym_id, leftmost_sym_obid


def getsortedsymbollists(traces, trace_ids, lbls, obj_ids):
    """
    sorts symbols from left to right using trace information
    :param traces: list of traces
    :param trace_ids: list of trace ids
    :param lbls: list of symbol labels
    :param obj_ids: list of symbol ids
    :return: sorted lists
    """
    traces_unsorted = traces.copy()
    trace_ids_unsorted = trace_ids.copy()
    lbls_unsorted = lbls.copy()
    obj_ids_unsorted = obj_ids.copy()

    traces_sorted = []
    trace_ids_sorted = []
    lbls_sorted = []
    obj_ids_sorted = []

    for oi in range(len(lbls)):
        if oi == 0:
            leftmost_sym_id, leftmost_sym_obid = getleftmostrootsymbol(traces_unsorted, obj_ids_unsorted, "BoundingBox")
        else:
            leftmost_sym_id, leftmost_sym_obid = getleftmostrootsymbol(traces_unsorted, obj_ids_unsorted, "BoundingBox")
        index_pop = obj_ids_unsorted.index(leftmost_sym_obid)
        index_append = obj_ids.index(leftmost_sym_obid)

        # Add sorted elemt
        traces_sorted.append(traces[index_append])
        trace_ids_sorted.append(trace_ids[index_append])
        lbls_sorted.append(lbls[index_append])
        obj_ids_sorted.append(obj_ids[index_append])

        # Remove from unsorted list
        traces_unsorted.pop(index_pop)
        trace_ids_unsorted.pop(index_pop)
        lbls_unsorted.pop(index_pop)
        obj_ids_unsorted.pop(index_pop)

    return traces_sorted, trace_ids_sorted, lbls_sorted, obj_ids_sorted


def getantirelation(relation):
    """
    returns antis of the relations
    :param relation: relation label
    :return: anti relation
    """
    antirelation = None
    if relation == "Sup": antirelation = "Sub"
    if relation == "Sub": antirelation = "Sup"
    if relation == "Above": antirelation = "Below"
    if relation == "Below": antirelation = "Above"
    return antirelation


def _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_expressions):
    """
    Recursively subdivides an expression into several sub-expressions.
    returns relationships of subexpressions that contain only two
    symbols
    :param traces: list of traces
    :param trace_ids: list of trace ids
    :param lbls: list of symbol labels
    :param obj_ids: list of symbol ids
    :param model: relationship classifier
    :param relations: list of relations in this expression
    :param sub_expressions: list of sub expressions
    :return: list of relations
    """
    for sub_ex in sub_expressions:
        switch_type = []
        root_index = []
        sub_exps = []
        above_first = None

        # Base case
        if len(sub_ex) < 2:
            continue

        # Further divide sub expressions
        for ii in range(len(sub_ex) -1):
            index1 = sub_ex[ii]
            if len(root_index) != 0:
                index1 = root_index[-1]
            index2 = sub_ex[ii+1]

            minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
            minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])

            # Predict relations
            feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
            predicted_relations = getpredictedrelationship(model, [feat])
            relation = predicted_relations[0][0]

            # Manually fix COMMA relations
            o_id1 = obj_ids[index1]
            o_id2 = obj_ids[index2]
            comma = ','
            if comma in o_id1 or comma in o_id2:
                relation = "Right"

            # Add relations of the root symbols
            # Add leaf symbols to the sub expressions
            if len(root_index) == 0:
                if relation != "Right":
                    switch_type.append(relation)
                    root_index.append(index1)
                    sub_exps.append([index2])
                    if relation == "Above":
                        above_first = True
                        sub_exps.append([])
                    elif relation == "Below":
                        above_first = False
                        sub_exps.append([])
                # Create LG format relationship
                o_id1 = o_id1.replace(comma, "COMMA")
                o_id2 = o_id2.replace(comma, "COMMA")
                relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
            else:
                if relation == "Above" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
                    if above_first:
                        sub_exps[-2].append(index2)
                    else:
                        if len(sub_exps[-1]) == 0:
                            # Create LG format relationship
                            o_id1 = o_id1.replace(comma, "COMMA")
                            o_id2 = o_id2.replace(comma, "COMMA")
                            relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                        sub_exps[-1].append(index2)

                elif relation == "Below" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
                    if above_first:
                        if len(sub_exps[-1]) == 0:
                            # Create LG format relationship
                            o_id1 = o_id1.replace(comma, "COMMA")
                            o_id2 = o_id2.replace(comma, "COMMA")
                            relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                        sub_exps[-1].append(index2)

                    else:
                        sub_exps[-2].append(index2)
                elif relation != "Right":
                    if len(sub_exps[-1]) == 0:
                        # Create LG format relationship
                        o_id1 = o_id1.replace(comma, "COMMA")
                        o_id2 = o_id2.replace(comma, "COMMA")
                        relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                    sub_exps[-1].append(index2)
                else:
                    switch_type.pop()
                    root_index.pop()
                    # Create LG format relationship
                    o_id1 = o_id1.replace(comma, "COMMA")
                    o_id2 = o_id2.replace(comma, "COMMA")
                    relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")

        # Recursive division
        if len(sub_exps) != 0:
            relations = _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_exps)

    return relations


def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
    """
    Subdivides an expression into several sub-expressions.
    returns relationships in an expression in OR format of
    label graph
    :param traces: list of traces
    :param trace_ids: list of trace ids
    :param lbls: list of symbol labels
    :param obj_ids: list of symbol ids
    :param model: relationship classifier
    :return: list of relations
    """

    # sort symbols based on min X i.e. left to right
    traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)

    relations = []
    switch_type = []
    root_index = []
    sub_expressions = []
    above_first = None

    # for consecutive pair of symbols
    for oi in range(len(lbls)-1):
        index1 = oi
        if len(root_index) != 0:
            index1 = root_index[-1]
        index2 = oi + 1

        minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
        minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])

        # Predict relations
        feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
        predicted_relations = getpredictedrelationship(model, [feat])
        relation = predicted_relations[0][0]

        # Manually fix COMMA relations
        o_id1 = obj_ids[index1]
        o_id2 = obj_ids[index2]
        comma = ','

        # Add relations of the root symbols
        # Add leaf symbols to the sub expressions
        if len(root_index) == 0:
            if relation != "Right":
                switch_type.append(relation)
                root_index.append(index1)
                sub_expressions.append([index2])
                if relation == "Above":
                    above_first = True
                    sub_expressions.append([])
                elif relation == "Below":
                    above_first = False
                    sub_expressions.append([])
            # Create LG format relationship
            o_id1 = o_id1.replace(comma, "COMMA")
            o_id2 = o_id2.replace(comma, "COMMA")
            relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
        else:
            if relation == "Above" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
                if above_first: sub_expressions[-2].append(index2)
                else:
                    if len(sub_expressions[-1]) == 0:
                        # Create LG format relationship
                        o_id1 = o_id1.replace(comma, "COMMA")
                        o_id2 = o_id2.replace(comma, "COMMA")
                        relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                    sub_expressions[-1].append(index2)

            elif  relation == "Below" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
                if above_first:
                    if len(sub_expressions[-1]) == 0:
                        # Create LG format relationship
                        o_id1 = o_id1.replace(comma, "COMMA")
                        o_id2 = o_id2.replace(comma, "COMMA")
                        relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                    sub_expressions[-1].append(index2)

                else: sub_expressions[-2].append(index2)
            elif relation != "Right":
                if len(sub_expressions[-1]) == 0:
                    # Create LG format relationship
                    o_id1 = o_id1.replace(comma, "COMMA")
                    o_id2 = o_id2.replace(comma, "COMMA")
                    relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
                sub_expressions[-1].append(index2)
            else:
                switch_type.pop()
                root_index.pop()
                # Create LG format relationship
                if comma in o_id1 or comma in o_id2:
                    relation = "Right"
                o_id1 = o_id1.replace(comma, "COMMA")
                o_id2 = o_id2.replace(comma, "COMMA")
                relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )

    # Send sub expressions to the recursive parsing function
    if len(sub_expressions) != 0:
        relations = _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_expressions)

    return relations


# def _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_expressions):
#
#     for sub_ex in sub_expressions:
#         switch_type = []
#         root_index = []
#         sub_exps = []
#         above_first = None
#         if len(sub_ex) < 2:
#             continue
#         for ii in range(len(sub_ex) -1):
#             index1 = sub_ex[ii]
#             if len(root_index) != 0:
#                 index1 = root_index[-1]
#             index2 = sub_ex[ii+1]
#
#             minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#             minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#             feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#             predicted_relations = getpredictedrelationship(model, [feat])
#             relation = predicted_relations[0][0]
#
#             # Manually fix COMMA relations
#             o_id1 = obj_ids[index1]
#             o_id2 = obj_ids[index2]
#             comma = ','
#             if comma in o_id1 or comma in o_id2:
#                 relation = "Right"
#
#             if len(root_index) == 0:
#                 if relation != "Right":
#                     switch_type.append(relation)
#                     root_index.append(index1)
#                     sub_exps.append([index2])
#                     if relation == "Above":
#                         above_first = True
#                         sub_exps.append([])
#                     elif relation == "Below":
#                         above_first = False
#                         sub_exps.append([])
#                 # Create LG format relationship
#                 o_id1 = o_id1.replace(comma, "COMMA")
#                 o_id2 = o_id2.replace(comma, "COMMA")
#                 relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#             else:
#                 if relation == "Above" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
#                     if above_first:
#                         sub_exps[-1].append(index2)
#                     else:
#                         if len(sub_exps[-2]) == 0:
#                             sub_exps[-2].append(index2)
#                             # Create LG format relationship
#                             o_id1 = o_id1.replace(comma, "COMMA")
#                             o_id2 = o_id2.replace(comma, "COMMA")
#                             relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#
#                 elif relation == "Below" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
#                     if above_first:
#                         if len(sub_exps[-2]) == 0:
#                             sub_exps[-2].append(index2)
#                             # Create LG format relationship
#                             o_id1 = o_id1.replace(comma, "COMMA")
#                             o_id2 = o_id2.replace(comma, "COMMA")
#                             relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#
#                     else:
#                         sub_exps[-1].append(index2)
#                 elif relation != "Right":
#                     if len(sub_exps[-1]) == 0:
#                         # Create LG format relationship
#                         o_id1 = o_id1.replace(comma, "COMMA")
#                         o_id2 = o_id2.replace(comma, "COMMA")
#                         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#                     sub_exps[-1].append(index2)
#                 else:
#                     switch_type.pop()
#                     root_index.pop()
#                     # Create LG format relationship
#                     o_id1 = o_id1.replace(comma, "COMMA")
#                     o_id2 = o_id2.replace(comma, "COMMA")
#                     relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#
#         if len(sub_exps) != 0:
#             relations = _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_exps)
#
#     return relations
#
#
# def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
#
#
#     # sort symbols based on min X i.e. left to right
#     traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)
#
#     relations = []
#     switch_type = []
#     root_index = []
#     sub_expressions = []
#     above_first = None
#     for oi in range(len(lbls)-1):
#         index1 = oi
#         if len(root_index) != 0:
#             index1 = root_index[-1]
#         index2 = oi + 1
#
#         minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#         minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#         feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#         predicted_relations = getpredictedrelationship(model, [feat])
#         relation = predicted_relations[0][0]
#
#         # Manually fix COMMA relations
#         o_id1 = obj_ids[index1]
#         o_id2 = obj_ids[index2]
#         comma = ','
#         # if comma in o_id1 or comma in o_id2:
#         #     relation = "Right"
#
#         if len(root_index) == 0:
#             if relation != "Right":
#                 switch_type.append(relation)
#                 root_index.append(index1)
#                 sub_expressions.append([index2])
#                 if relation == "Above":
#                     above_first = True
#                     sub_expressions.append([])
#                 elif relation == "Below":
#                     above_first = False
#                     sub_expressions.append([])
#             # Create LG format relationship
#             o_id1 = o_id1.replace(comma, "COMMA")
#             o_id2 = o_id2.replace(comma, "COMMA")
#             relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#         else:
#             if relation == "Above" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
#                 if above_first: sub_expressions[-1].append(index2)
#                 else:
#                     if len(sub_expressions[-2]) == 0:
#                         sub_expressions[-2].append(index2)
#                         # Create LG format relationship
#                         o_id1 = o_id1.replace(comma, "COMMA")
#                         o_id2 = o_id2.replace(comma, "COMMA")
#                         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#
#             elif  relation == "Below" and (switch_type[-1] == "Above" or switch_type[-1] == "Below"):
#                 if above_first:
#                     if len(sub_expressions[-2]) == 0:
#                         sub_expressions[-2].append(index2)
#                         # Create LG format relationship
#                         o_id1 = o_id1.replace(comma, "COMMA")
#                         o_id2 = o_id2.replace(comma, "COMMA")
#                         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#
#                 else: sub_expressions[-1].append(index2)
#             elif relation != "Right":
#                 if len(sub_expressions[-1]) == 0:
#                     # Create LG format relationship
#                     o_id1 = o_id1.replace(comma, "COMMA")
#                     o_id2 = o_id2.replace(comma, "COMMA")
#                     relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0")
#                 sub_expressions[-1].append(index2)
#             else:
#                 switch_type.pop()
#                 root_index.pop()
#                 # Create LG format relationship
#                 if comma in o_id1 or comma in o_id2:
#                     relation = "Right"
#                 o_id1 = o_id1.replace(comma, "COMMA")
#                 o_id2 = o_id2.replace(comma, "COMMA")
#                 relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#
#     if len(sub_expressions) != 0:
#         relations = _expressionparser(traces, trace_ids, lbls, obj_ids, model, relations, sub_expressions)
#
#     return relations


# def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
#
#     relations = []
#     visited = []
#     if len(lbls) != len(obj_ids):
#         print(filename)
#     # sort symbols based on min X i.e. left to right
#     traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)
#
#     usepreviousindex1 = False
#     prev_index1 = -1
#     for oi in range(len(lbls)-1):
#         if not usepreviousindex1:
#             index1 = oi
#             prev_index1 = oi
#         else:
#             index1 = prev_index1
#         index2 = oi + 1
#
#         minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#         minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#         feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#         predicted_relations = getpredictedrelationship(model, [feat])
#         relation = predicted_relations[0][0]
#
#         # Manually fix COMMA relations
#         o_id1 = obj_ids[index1]
#         o_id2 = obj_ids[index2]
#         comma = ','
#         if comma in o_id1 or comma in o_id2:
#             relation = "Right"
#
#         if relation != "Right":
#             usepreviousindex1 = True
#         else:
#             usepreviousindex1 = False
#
#         # Create LG format relationship
#         # o_id1 = obj_ids[index1]
#         # o_id2 = obj_ids[index2]
#         # comma = ','
#         # if comma in o_id1 or comma in o_id2:
#         #     relation ="Right"
#         o_id1 = o_id1.replace(comma, "COMMA")
#         o_id2 = o_id2.replace(comma, "COMMA")
#         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#
#     return relations


# def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
#     """
#     Uses immediate next
#     :param traces:
#     :param trace_ids:
#     :param lbls:
#     :param obj_ids:
#     :param filename:
#     :param model:
#     :return:
#     """
#
#     relations = []
#     # sort symbols based on min X i.e. left to right
#     traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)
#
#     baseline_exit = False
#     switch_type = None
#     baseline_index = None
#     for oi in range(len(lbls)-1):
#         index1 = oi
#         index2 = oi + 1
#
#         minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#         minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#         feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#         predicted_relations = getpredictedrelationship(model, [feat])
#         relation = predicted_relations[0][0]
#
#         if baseline_exit == False:
#             if relation != "Right" and relation != "Inside":
#                 baseline_exit = True
#                 switch_type = relation
#                 baseline_index = index1
#         else:
#             if relation == getantirelation(switch_type):
#                 baseline_exit = False
#                 index1 = baseline_index
#                 if relation != "Above" and relation != "Below":
#                     relation = "Right"
#
#         # Create LG format relationship
#         o_id1 = obj_ids[index1]
#         o_id2 = obj_ids[index2]
#         comma = ','
#         if comma in o_id1 or comma in o_id2:
#             relation ="Right"
#         o_id1 = o_id1.replace(comma, "COMMA")
#         o_id2 = o_id2.replace(comma, "COMMA")
#
#         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#
#     return relations


# def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
#     """
#     Uses immediate next
#     :param traces:
#     :param trace_ids:
#     :param lbls:
#     :param obj_ids:
#     :param filename:
#     :param model:
#     :return:
#     """
#
#     relations = []
#     # sort symbols based on min X i.e. left to right
#     traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)
#
#     baseline_exit = []
#     switch_type = []
#     baseline_index = []
#     for oi in range(len(lbls)-1):
#         index1 = oi
#         index2 = oi + 1
#
#         minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#         minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#         feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#         predicted_relations = getpredictedrelationship(model, [feat])
#         relation = predicted_relations[0][0]
#
#         # Manually fix COMMA relations
#         o_id1 = obj_ids[index1]
#         o_id2 = obj_ids[index2]
#         comma = ','
#         if comma in o_id1 or comma in o_id2:
#             relation = "Right"
#
#
#         if len(baseline_exit) == 0:
#             if relation != "Right" and relation != "Inside":
#                 baseline_exit.append(1)
#                 switch_type.append(relation)
#                 baseline_index.append(index1)
#         else:
#             if relation == getantirelation(switch_type[-1]):
#                 expected = None
#                 if relation == "Sup" or relation =="Sub":
#                     expected = "Right"
#                 elif relation == "Above":
#                     expected = "Above"
#                 elif relation == "Below":
#                     expected = "Below"
#                 new_rel = None
#                 while new_rel != expected and len(baseline_exit) > 0:
#                     baseline_exit.pop()
#                     switch_type.pop()
#                     index1 = baseline_index.pop()
#
#                     minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#                     minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#                     feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#                     predicted_relations = getpredictedrelationship(model, [feat])
#                     new_rel = predicted_relations[0][0]
#                 relation = new_rel
#                 # if relation != "Above" and relation != "Below":
#                 #     relation = "Right"
#             elif relation != "Right" and relation != "Inside":
#                 baseline_exit.append(1)
#                 switch_type.append(relation)
#                 baseline_index.append(index1)
#
#         # Create LG format relationship
#         # o_id1 = obj_ids[index1]
#         # o_id2 = obj_ids[index2]
#         # comma = ','
#         # if comma in o_id1 or comma in o_id2:
#         #     relation ="Right"
#         o_id1 = o_id1.replace(comma, "COMMA")
#         o_id2 = o_id2.replace(comma, "COMMA")
#
#         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#
#     return relations


# def expressionparser(traces, trace_ids, lbls, obj_ids, filename, model):
#     """
#     Uses immediate next
#     :param traces:
#     :param trace_ids:
#     :param lbls:
#     :param obj_ids:
#     :param filename:
#     :param model:
#     :return:
#     """
#
#     relations = []
#     # sort symbols based on min X i.e. left to right
#     traces, trace_ids, lbls, obj_ids = getsortedsymbollists(traces, trace_ids, lbls, obj_ids)
#
#     baseline_exit = []
#     switch_type = []
#     baseline_index = []
#     for oi in range(len(lbls)-1):
#         index1 = oi
#         index2 = oi + 1
#
#         minx1, miny1, maxx1, maxy1 = getminmaxofXandY(traces[index1])
#         minx2, miny2, maxx2, maxy2 = getminmaxofXandY(traces[index2])
#
#         feat = getcornertocornervectorfeatures(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2)
#         predicted_relations = getpredictedrelationship(model, [feat])
#         relation = predicted_relations[0][0]
#
#         if len(baseline_exit) == 0:
#             if relation != "Right" and relation != "Inside":
#                 baseline_exit.append(1)
#                 switch_type.append(relation)
#                 baseline_index.append(index1)
#         else:
#             if relation == getantirelation(switch_type[-1]):
#                 baseline_exit.pop()
#                 switch_type.pop()
#                 index1 = baseline_index.pop()
#                 if relation != "Above" and relation != "Below":
#                     relation = "Right"
#             elif relation != "Right" and relation != "Inside":
#                 baseline_exit.append(1)
#                 switch_type.append(relation)
#                 baseline_index.append(index1)
#
#         # Create LG format relationship
#         o_id1 = obj_ids[index1]
#         o_id2 = obj_ids[index2]
#         comma = ','
#         if comma in o_id1 or comma in o_id2:
#             relation ="Right"
#         o_id1 = o_id1.replace(comma, "COMMA")
#         o_id2 = o_id2.replace(comma, "COMMA")
#
#         relations.append("R, " + o_id1 + ", " + o_id2 + ", " + relation + ", 1.0" )
#
#     return relations


def bulkparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test, model):
    """
    produces bulk parsing results
    :param listoftraces_test: list of traces
    :param listoftraceids_test: list of trace ids
    :param labels_test: list of symbol labels
    :param objectids_test: list of object ids
    :param filenames_test: list of filenames
    :param model: classifier model
    :return: Predicted expressions
    """
    parsed_exps = []
    for oi in range(len(filenames_test)):
        rels = expressionparser(listoftraces_test[oi], listoftraceids_test[oi],
                                labels_test[oi], objectids_test[oi], filenames_test[oi], model)
        parsed_exps.append(rels)
    return parsed_exps


def gettrainedrelationshipclassifier(featurevector_train, data_train_labels):
    """
    Trains a Rforest model and returns it
    :param featurevector_train: feature vector train
    :param data_train_labels: train data labels
    :return: RForest model
    """
    # Train a model for random forest and create its pickle
    if os.path.isfile(FILENAME_REL):
        return joblib.load(FILENAME_REL)
    else:
        random_forest = RandomForestClassifier(bootstrap=True,
                                                class_weight=None, criterion='gini',n_estimators=50,max_depth=50)
        random_forest.fit(featurevector_train, data_train_labels)
        filename = FILENAME_REL

        with open(filename, 'wb') as f:
            joblib.dump(random_forest, f)
        return random_forest


def getminmaxofXandY(raw_feature):
    """
    Gets minimum and maximum of X and Y
    from trace info of symbols
    :param raw_feature:
    :return:
    """
    localminX = 999999.0
    localmaxX = -999999.0
    localminY = 999999.0
    localmaxY = -999999.0
    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        if len(Xcoords) == 1:
            return Xcoords[0], Ycoords[0], Xcoords[0], Ycoords[0]
        for innerind in range(len(Xcoords) - 1):
            if Xcoords[innerind] < localminX:
                localminX = Xcoords[innerind]
            if Xcoords[innerind] > localmaxX:
                localmaxX = Xcoords[innerind]
            if Ycoords[innerind] < localminY:
                localminY = Ycoords[innerind]
            if Ycoords[innerind] > localmaxY:
                localmaxY = Ycoords[innerind]

    return localminX, localminY, localmaxX, localmaxY


def getcornertocornervectorfeatures(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2):
    """
    produces 9 unique features and returns its vector
    :param min_x1: min x symbol 1
    :param min_y1: min y symbol 1
    :param max_x1: max x symbol 1
    :param max_y1: max y symbol 1
    :param min_x2: min x symbol 2
    :param min_y2: min y symbol 2
    :param max_x2: max x symbol 2
    :param max_y2: max y symbol 2
    :return: feature vector
    """

    # Corner to corner
    top_right = math.degrees(math.atan2(max_y2 - max_y1, max_x2 - max_x1))
    top_left = math.degrees(math.atan2(max_y2 - max_y1, min_x2 - min_x1))
    bottom_right = math.degrees(math.atan2(min_y2 - min_y1, max_x2 - max_x1))
    bottom_left = math.degrees(math.atan2(min_y2 - min_y1, min_x2 - min_x1))

    centerX1 = (min_x1 + max_x1)/2
    centerX2 = (min_x2 + max_x2)/2
    centerY1 = (min_y1 + max_y1)/2
    centerY2 = (min_y2 + max_y2)/2

    # midpoint to midpoint
    top_mid = math.degrees(math.atan2(max_y2 - max_y1, centerX2 - centerX1))
    left_mid = math.degrees(math.atan2(centerY2 - centerY1, min_x2 - min_x1))
    bot_mid = math.degrees(math.atan2(min_y2 - min_y1, centerX2 - centerX1))
    right_mid = math.degrees(math.atan2(centerY2 - centerY1, max_x2 - max_x1))

    # center to center
    bbcenter = math.degrees(math.atan2(centerY2 - centerY1, centerX2 - centerX1))

    return [top_right, top_left, bottom_right, bottom_left,
            top_mid, left_mid, bot_mid, right_mid,
            bbcenter]


def gettracesfromobjectids(trcs, obj_ids, id1, id2):
    """
    given a two object ids, this function returns
    their corresponding trace information
    :param trcs: trace
    :param obj_ids: list of object ids
    :param id1: id1
    :param id2: id2
    :return: two traces
    """
    index1 = -1
    index2 = -1
    for i in range(len(obj_ids)):
        if index1 == -1 or index2 == -1:
            if obj_ids[i] == id1:
                index1 = i
            if obj_ids[i] == id2:
                index2 = i
        else:
            break;
    return trcs[index1], trcs[index2]


def extractrelationshipfeatures(traces, objectids, relations):
    """
    extracts relationship features for training classifier
    :param traces: list of traces
    :param objectids: list of object ids
    :param relations: list of relation labels
    :return: feature vector and GT labels
    """
    feature_vector = []
    rel_labels = []
    for oi in range(len(relations)):
        rels = relations[oi]
        for ii in range(len(rels)):
            sym_id_1 = rels[ii][0]
            sym_id_2 = rels[ii][1]
            rel_lbl = rels[ii][2]
            traces_1, traces_2 = gettracesfromobjectids(traces[oi], objectids[oi], sym_id_1, sym_id_2)
            minX1, minY1, maxX1, maxY1 = getminmaxofXandY(traces_1)
            minX2, minY2, maxX2, maxY2 = getminmaxofXandY(traces_2)
            feat = getcornertocornervectorfeatures(minX1, minY1, maxX1, maxY1, minX2, minY2, maxX2, maxY2)
            feature_vector.append(feat)
            rel_labels.append(rel_lbl)

    return feature_vector, rel_labels


def readinputfiles(filename):
    '''
    reads a textfile and returns the content in form of a list
    :param filename: name of the file without extention.
    :return: list of messages
    '''
    parts =[]
    try:
        with open(filename) as filedata:
            for line in filedata:
                parts.append(line.strip())
            return parts
    except FileNotFoundError:
        print("That's not a file! Run the program again!")
        return None


def readrelationsfromLGfiles(subdir, filenames):
    """
    reads relation data from corressponding LG file
    of the INKML file
    :param subdir: directory
    :param filenames: INKML filenames
    :return: relations list
    """
    relations = []
    for filename in filenames:
        if filename.endswith('.inkml'):
            fn = getjustfilenamewithoutpathorextension(filename)
            filecontent = readinputfiles(subdir + "\\" + fn + ".lg")
            onefilerelations = []
            for line in filecontent:
                if len(line) != 0:
                    if line[0] == 'R':
                        parts = line.split(',')
                        onefilerelations.append([ parts[1].strip(), parts[2].strip(), parts[3].strip()])
            relations.append(onefilerelations)
    return relations


def readinkmlfileswithobjectids(fn, listoftraces, listoftraceids, labels, objectids, filenames):
    """
    Reads inkml files using Elementtree
    :param fn: filename
    :param listoftraces: empty list of traces
    :param listoftraceids: empty list of trace ids
    :param labels: empty list of labels
    :param filenames: empty list of filenames
    :return: listoftraces, listoftraceids, labels, filenames
    """
    # print(fn)
    root = ET.parse(fn).getroot()
    onefilelabels = []
    onefileobjectids = []
    onefilelistoftraces = []
    onefilelistoftraceids = []
    for subel in root:
        if subel.tag == '{http://www.w3.org/2003/InkML}traceGroup':
            for subsubel in subel.getchildren():
                if subsubel.tag == '{http://www.w3.org/2003/InkML}traceGroup':
                    featureentry = []
                    traceidlist = []
                    for subsubsubel in subsubel.getchildren():
                        if subsubsubel.tag == '{http://www.w3.org/2003/InkML}annotation':
                            onefilelabels.append(subsubsubel.text)
                        if subsubsubel.tag == '{http://www.w3.org/2003/InkML}annotationXML':
                            onefileobjectids.append(subsubsubel.items()[0][1])
                        if subsubsubel.tag == '{http://www.w3.org/2003/InkML}traceView':
                            featureentry, traceidlist = gettraceinfo(root, featureentry, traceidlist, subsubsubel.items()[0][1])
                    onefilelistoftraces.append(featureentry)
                    onefilelistoftraceids.append(traceidlist)
    if listoftraces != None:
        listoftraces.append(onefilelistoftraces)
        listoftraceids.append(onefilelistoftraceids)
        labels.append(onefilelabels)
        objectids.append(onefileobjectids)
        filenames.append(fn)
        return listoftraces, listoftraceids, labels, objectids, filenames
    else:
        return onefilelistoftraces, onefilelistoftraceids, onefilelabels, onefileobjectids, fn


def readdirectoryforinkmlfileswithobjectids(subdir, listoftraces, listoftraceids, labels, objectids, filenames):
    """
    reads subdirectories in a directory and inkml files in them
    :param subdir: subdirectory
    :param listoftraces: empty list of traces
    :param listoftraceids: empty list of trace ids
    :param labels: empty list of labels
    :param filenames: empty list of filenames
    :return: listoftraces, listoftraceids, labels, filenames
    """
    filelist = os.listdir(subdir)
    for item in filelist:
        if item.endswith('.inkml'):
            listoftraces, listoftraceids, labels, objectids, filenames = \
                readinkmlfileswithobjectids(subdir + "\\" +item, listoftraces, listoftraceids, labels, objectids, filenames)
    return listoftraces, listoftraceids, labels, objectids, filenames


def bulksmoothing(listoftraces):
    """
    smooths traces
    :param listoftraces: list of traces
    :return: smooth traces
    """
    for outer_ind in range(len(listoftraces)):
        for inner_ind in range(len(listoftraces[outer_ind])):
            listoftraces[outer_ind][inner_ind] = smoothdatapointsandinterpolate(
                removeconsecutiveduplicatepoints(listoftraces[outer_ind][inner_ind]))
    return listoftraces


def mapsegmentationtooriginalstrokedata(orig_traces, orig_traceids, bulk_segm):
    """
    Gets corresponding segmentation strokes for stroke level
    parsing
    :param orig_traces: original traces
    :param orig_traceids: original trace ids
    :param bulk_segm: their segmentation
    :return: traces based on the segmentation
    """
    orig_segm_strokes = []
    for ooi in range(len(bulk_segm)):
        strokes_flat, strokeids_flat = flattentraces(orig_traces[ooi], orig_traceids[ooi])
        symbols = bulk_segm[ooi]
        orig_symbols = []
        for oi in range(len(symbols)):
            sym = symbols[oi]
            orig_sym_strokes = []
            for ii in range(len(sym)):
                index = strokeids_flat.index(sym[ii])
                orig_sym_strokes.append(strokes_flat[index][0])
                orig_sym_strokes.append(strokes_flat[index][1])
            orig_symbols.append(orig_sym_strokes)
        orig_segm_strokes.append(orig_symbols)
    return orig_segm_strokes


def separateXandYcoordlists(bulk_strokes):
    """
    saves X and Y coordinates in separate lists
    :param bulk_strokes: list of traces
    :return: X and Y separated list of list
    """
    for ooi in range(len(bulk_strokes)):
        symbols = bulk_strokes[ooi]
        for oi in range(len(symbols)):
            coordlist = []
            sym = symbols[oi]
            for ii in range(len(sym)):
                coordlist.append(sym[ii][0])
                coordlist.append(sym[ii][1])
            symbols[oi] = coordlist
    return bulk_strokes


def getfakeobjectids(listofobjectids, listoflabels):
    """
    creates fake object ids for bonus
    :param listofobjectids: list of object ids
    :param listoflabels: list of labels
    :return: fake random object ids
    """
    for oi in range(len(listoflabels)):
        oids = []
        for ii in range(len(listoflabels[oi])):
            oids.append("AUTO_"+str(ii))
        listofobjectids[oi] = oids
    return listofobjectids


def strokeparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test):
    """
    Does stroke level parsing of data
    :param listoftraces_test: list of traces
    :param listoftraceids_test: list of trace ids
    :param labels_test: list of labels
    :param objectids_test: list of object ids
    :param filenames_test: list of filenames
    :return: parsed expressions
    """
    print("STROKE LEVEL PARSING")

    print("Extracting features and creating a relationship classifier.")
    model = gettrainedrelationshipclassifier(None, None)
    recognizer = gettrainedrandomforestmodel(None, None)

    print("Performing Segmentation and creating feature vectors of the segments.")
    bulk_segmentation_strokes, bulk_segmentation_strokeids, bulk_featurevectors = \
        producebulktestfeaturevectoranddosegmentation(deepcopy(listoftraces_test), listoftraceids_test, labels_test, filenames_test)

    print("Preprocessing the data.")
    listoftraces_test = bulksmoothing(listoftraces_test)
    # listoftraces_test = separateXandYcoordlists(listoftraces_test)
    pred_segm_orig_strokes = mapsegmentationtooriginalstrokedata(listoftraces_test, listoftraceids_test, bulk_segmentation_strokeids)

    print("Running Recognizer (Classifier) on the created segments.")
    all_expressions = performbulkclassification(recognizer, bulk_featurevectors)

    objids_predicted = []
    print("Bulk parsing")
    for i in range(len(filenames_test)):
        o_ids = []
        for j in range(len(all_expressions[i])):
            o_ids.append("obj_"+str(j))
        objids_predicted.append(o_ids)
    parse_results = bulkparsing(pred_segm_orig_strokes, bulk_segmentation_strokeids, all_expressions, objids_predicted, filenames_test, model)
    print("Stroke level parsing is done. Generating LG files")

    return filenames_test, objids_predicted, all_expressions, bulk_segmentation_strokeids, parse_results


def symbolparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test):
    """
    Uses ground truth segmentation and classification and parses the data
    :param listoftraces_test: list of traces
    :param listoftraceids_test: list of trace ids
    :param labels_test: list of labels
    :param objectids_test: list of object ids
    :param filenames_test: list of filenames
    :return: parsed expressions
    """
    print("SYMBOL LEVEL PARSING")

    print("Preprocessing the data.")
    listoftraces_test = bulksmoothing(listoftraces_test)

    print("Extracting features and creating a relationship classifier.")
    # fv_rel_train, rel_lbls_train = extractrelationshipfeatures(listoftraces_train, objectids_train, relations_train)
    # model = gettrainedrelationshipclassifier(fv_rel_train, rel_lbls_train )
    model = gettrainedrelationshipclassifier(None, None)

    print("Bulk parsing")
    parse_results = bulkparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test, model)
    print("Symbol level parsing is done. Generating LG files")

    return parse_results


def doonetimeparsingstuff():
    """
    Things that needs to be done just once
    Saves a trained random forest model as a pickle
    """

    print("Reading split data.")
    # listoftraces_train, listoftraceids_train, labels_train, objectids_train, filenames_train = \
    #     readdirectoryforinkmlfileswithobjectids(testtraininkmlpath, [], [], [], [], [])
    # relations_train = readrelationsfromLGfiles(testtrainlgpath, filenames_train)
    #
    # fv_rel_train, rel_lbls_train = extractrelationshipfeatures(listoftraces_train, objectids_train, relations_train)
    # model = gettrainedrelationshipclassifier(fv_rel_train, rel_lbls_train )

    listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test = \
        readdirectoryforinkmlfileswithobjectids(testtraininkmlpath, [], [], [], [], [])
    # relations_test = readrelationsfromLGfiles(testtestlgpath, filenames_test)

    listoftraces_test_copy = deepcopy(listoftraces_test)

    # For symbol level parsing
    # parse_results = symbolparsing(listoftraces_test_copy, listoftraceids_test, labels_test, objectids_test, filenames_test)
    # producebulkparsedLGfiles(filenames_test, objectids_test, labels_test, listoftraceids_test, parse_results,
    #                          predictedTestLG_Symbols)

    # For stroke level parsing
    filenames_test, objids_predicted, all_expressions, bulk_segmentation_strokeids, parse_results = \
        strokeparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test)
    producebulkparsedLGfiles(filenames_test, objids_predicted, all_expressions, bulk_segmentation_strokeids, parse_results, predictedTrainLG_Strokes)

    print("One time parsing is done!")


def bonusparsing(symORstr):
    """
    parsing bonus dataset
    :param symORstr: symbol level or stroke level
    :return: produces LG files
    """
    print("Bonus parsing, started")
    print("Reading bonus data.")
    listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test = \
        readdirectoryforinkmlfileswithobjectids(bonusdatapath, [], [], [], [], [])
    objectids_test = getfakeobjectids(objectids_test, labels_test)

    # For symbol level parsing
    listoftraces_test_copy = deepcopy(listoftraces_test)
    parse_results = symbolparsing(listoftraces_test_copy, listoftraceids_test, labels_test, objectids_test, filenames_test)
    producebulkparsedLGfiles(filenames_test, objectids_test, labels_test, listoftraceids_test, parse_results,
                             predictedBonusLG_Symbols)

    print("Bonus parsing, ended")


def parselistofinkmlfromtxtfile(txtpath, symORstr):
    """
    Given an input textfile with list of INKML files,
    generates LG files of predicted expressions
    :param txtpath: textfile pth
    :param symORstr: symbl or strok level
    :return: produces LG files
    """
    print("Recognizing expressions from a list of INKML files")
    txtfilename = getjustfilenamewithoutpathorextension(txtpath)
    filenames = readinputfiles(txtpath)

    print("Reading data.")
    listoftraces_test = []
    listoftraceids_test = []
    labels_test = []
    objectids_test = []
    filenames_test = []
    for fn in filenames:
        listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test = \
            readinkmlfileswithobjectids(fn, listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test)

    if symORstr == "1" or symORstr == 1:
        # For symbol level parsing
        parse_results = symbolparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test)
        lgfilessavedirectory = txtpath[:txtpath.rfind('\\')] + "\\" + txtfilename + "_symbollevel"
        print("Output folder for input txtfile: " +txtfilename+".txt created in the same folder by the name: " + txtfilename + "_symbollevel")
        if not os.path.exists(lgfilessavedirectory):
            os.makedirs(lgfilessavedirectory)
        producebulkparsedLGfiles(filenames_test, objectids_test, labels_test, listoftraceids_test, parse_results,
                                 lgfilessavedirectory)

    if symORstr == "2" or symORstr == 2:
        # For stroke level parsing
        filenames_test, objids_predicted, all_expressions, bulk_segmentation_strokeids, parse_results = \
            strokeparsing(listoftraces_test, listoftraceids_test, labels_test, objectids_test, filenames_test)
        lgfilessavedirectory = txtpath[:txtpath.rfind('\\')] + "\\" + txtfilename + "_strokelevel"
        print("Output folder for input txtfile: " +txtfilename+".txt created in the same folder by the name: " + txtfilename + "_strokelevel")
        if not os.path.exists(lgfilessavedirectory):
            os.makedirs(lgfilessavedirectory)
        producebulkparsedLGfiles(filenames_test, objids_predicted, all_expressions, bulk_segmentation_strokeids, parse_results, lgfilessavedirectory)

    return None


def main():
    """
    main method
    """
    print("STARTED PARSING")
    # doonetimeparsingstuff()
    # bonusparsing("1")
    parselistofinkmlfromtxtfile( sys.argv[1] , sys.argv[2])
    # parselistofinkmlfromtxtfile("D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project3 - Parsing\TestProject\.idea\\testtxtfile1.txt", "2")
    print("PARSING - THE END!")


if __name__ == "__main__":
    main()