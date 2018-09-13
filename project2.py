"""
project2.py

Authors: Dhaval Chauhan dmc8686@g.rit.edu
Authors: Bhavin Bhuta bsb5375@g.rit.edu

Preprocess CROHME dataset, extract features and do
classification and segmentation of symbols using
random forest.
"""

import xml.etree.ElementTree as ET
from pylab import *
import numpy as np
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


FILENAME = "randomforest.sav"
MERGINGTHRESHFACTOR = 1.0
MINPROBREQFORMERGING = 0.6
CLOSESTTWOPOINTS = "closesttwopoints"
CENTEROFGRAVITY = "centerofgravity"



def getmergingthreshold( minX, maxX, minY, maxY, numberofstrokes ):
    """
    Calculates a merging threshold based on the number of input
    primitives in the expression and a constant user controlled
    factor.
    :param minX: minimum X
    :param maxX: maximum X
    :param minY: minimum Y
    :param maxY: Maximum Y
    :param numberofstrokes: number of strokes in the expression
    :return: calculated merging threshold
    """
    maxrange = 0.0
    minrange = 0.0
    if maxX-minX <= maxY - minY:
        maxrange = maxY - minY
        minrange = maxX - minX
    else:
        maxrange = maxX - minX
        minrange = maxY - minY
    return MERGINGTHRESHFACTOR*(maxrange/(numberofstrokes+1))


def getminmaxXYinexpression(allstrokes):
    """
    Gets minimum and maximum X and Y values from the stroke information
    :param allstrokes: list of all strokes in an expression
    :return: minX, maxX, minY, minY
    """
    localminX = 999999.0
    localmaxX = -999999.0
    localminY = 999999.0
    localmaxY = -999999.0
    for ind in range(len(allstrokes)):
        Xcoords = allstrokes[ind][0]
        Ycoords = allstrokes[ind][1]
        # Find local mins and maxs
        for ind_in_coord in range(len(Xcoords)):
            if Xcoords[ind_in_coord] < localminX:
                localminX = Xcoords[ind_in_coord]
            if Xcoords[ind_in_coord] > localmaxX:
                localmaxX = Xcoords[ind_in_coord]
            if Ycoords[ind_in_coord] < localminY:
                localminY = Ycoords[ind_in_coord]
            if Ycoords[ind_in_coord] > localmaxY:
                localmaxY = Ycoords[ind_in_coord]
    return localminX, localmaxX, localminY, localmaxY


def flattentraces(listoftraces, listoftraceids):
    """
    Puts all the traces into one list to toss
    away its ground truth information
    :param listoftraces: list of list of traces
    :param listoftraceids: list of list of trace ids
    :return: list of traces and their ids
    """
    strokes = []
    strokeids = []
    for outer_ind in range(len(listoftraces)):
        for inner_ind in range(len(listoftraces[outer_ind])//2):
            strokes.append([listoftraces[outer_ind][2*inner_ind], listoftraces[outer_ind][2*inner_ind + 1]])
            strokeids.append(listoftraceids[outer_ind][inner_ind])
    return strokes, strokeids


def calculatecenterofgravity(stroke):
    """
    calculates mean of X and Y values of points
    :param stroke: list of strokes
    :return: Xmean and Ymean
    """
    sumX = 0
    sumY = 0
    n = len(stroke[0])
    for ind in range(n):
        sumX += stroke[0][ind]
        sumY += stroke[1][ind]
    return [sumX/n, sumY/n]


def calculateminimumtwopointdistance(stroke1, stroke2):
    """
    Finds the closest two points of the two strokes and
    returns the distance between those two points
    :param stroke1: first stroke
    :param stroke2: second stroke
    :return: minimum distance between the two strokes
    """
    mindist = 999999.0
    for ind1 in range(len(stroke1[0])):
        for ind2 in range(len(stroke2[0])):
            dist = math.sqrt((stroke1[0][ind1] - stroke2[0][ind2])**2 + (stroke1[1][ind1] - stroke2[1][ind2])**2)
            if dist < mindist:
                mindist = dist
    return mindist


def checkiftwostrokescanbemergedornot(stroke1, stroke2, mthresh, criterion):
    """
    checks if two strokes can be merged or not based on the criterion
    and the threshold.
    :param stroke1: first stroke
    :param stroke2: second stroke
    :param mthresh: maximum distance threshold
    :param criterion: either center of gravity or closest two points
    :return: boolean yes or not
    """
    if criterion == "centerofgravity":
        center1 = calculatecenterofgravity(stroke1)
        center2 = calculatecenterofgravity(stroke2)
        if mthresh >= math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2):
            return True
        return False
    elif criterion == "closesttwopoints":
        if mthresh >= calculateminimumtwopointdistance(stroke1, stroke2):
            return True
        return False


def resegment(sym, sym_id):
    """
    finds two traces with largest closest point
    distance and then creates a split between
    those two symbols
    :param sym: list of traces in a symbol
    :param sym_id: list of trace ids in a symbol
    :return: list of new symbols and their strokes
    """
    new_syms = []
    new_sym_ids = []
    dists = []
    for ind in range(len(sym) - 1):
        dists.append(calculateminimumtwopointdistance(sym[ind], sym[ind+1]))
    maxdist_index = dists.index(max(dists))
    new_syms.append(sym[:maxdist_index+1])
    new_sym_ids.append(sym_id[:maxdist_index+1])
    new_syms.append(sym[maxdist_index+1:])
    new_sym_ids.append(sym_id[maxdist_index+1:])

    return new_syms, new_sym_ids


def performbaselinesegmentation(segmentation_strokes, segmentation_strokeids):
    """
    splits the segmentation set that contains more than 3 traces into
    two segmentation sets.
    :param segmentation_strokes: input strokes
    :param segmentation_strokeids: input stroke ids
    :return: output strokes and ids after split
    """
    expr = segmentation_strokes
    expr_id = segmentation_strokeids
    new_expr = []
    new_expr_id = []
    for sym_ind in range(len(expr)):
        sym = expr[sym_ind]
        sym_id = expr_id[sym_ind]
        if len(sym) > 3:
            new_syms, new_sym_ids = resegment(sym, sym_id)
            for ind in range(len(new_syms)):
                new_expr.append(new_syms[ind])
                new_expr_id.append(new_sym_ids[ind])
        else:
            new_expr.append(sym)
            new_expr_id.append(sym_id)

    return new_expr, new_expr_id


def performnonbaselinesegmentation(allstrokes, allstrokeids, mthresh):
    """
    Given a list of strokes, this function creates a bunch of sets
    indicating the strokes which can be merged together to form a
    symbol.
    :param allstrokes: list of strokes
    :param allstrokeids: list of trace ids of strokes
    :param mthresh: maximum distance threshold
    :return: segmentation strokes and their ids
    """
    segmentation_strokes = []
    segmentation_strokeids = []
    segmentation_strokelistids = []
    for ind1 in range(len(allstrokes) - 1):
        if checkiftwostrokescanbemergedornot(allstrokes[ind1], allstrokes[ind1 + 1], mthresh, CENTEROFGRAVITY):
            if len(segmentation_strokelistids) != 0:
                if ind1 in segmentation_strokelistids[len(segmentation_strokelistids) - 1]:
                    segmentation_strokes[len(segmentation_strokes) - 1].append(allstrokes[ind1 + 1])
                    segmentation_strokeids[len(segmentation_strokes) - 1].append(allstrokeids[ind1 + 1])
                    segmentation_strokelistids[len(segmentation_strokes) - 1].append(ind1 + 1)
                else:
                    segmentation_strokes.append([allstrokes[ind1], allstrokes[ind1 + 1]])
                    segmentation_strokeids.append([allstrokeids[ind1], allstrokeids[ind1 + 1]])
                    segmentation_strokelistids.append([ind1, ind1 + 1])
            else:
                segmentation_strokes.append([allstrokes[ind1], allstrokes[ind1 + 1]])
                segmentation_strokeids.append([allstrokeids[ind1], allstrokeids[ind1 + 1]])
                segmentation_strokelistids.append([ind1, ind1 + 1])
        else:
            if len(segmentation_strokelistids) != 0:
                if not ind1 in segmentation_strokelistids[len(segmentation_strokelistids) - 1]:
                    segmentation_strokes.append([allstrokes[ind1]])
                    segmentation_strokeids.append([allstrokeids[ind1]])
                    segmentation_strokelistids.append([ind1])
            else:
                segmentation_strokes.append([allstrokes[ind1]])
                segmentation_strokeids.append([allstrokeids[ind1]])
                segmentation_strokelistids.append([ind1])

    if len(segmentation_strokelistids) != 0:
        if not (len(allstrokeids) - 1) in segmentation_strokelistids[len(segmentation_strokelistids) - 1]:
            segmentation_strokes.append([allstrokes[(len(allstrokeids) - 1)]])
            segmentation_strokeids.append([allstrokeids[(len(allstrokeids) - 1)]])
            segmentation_strokelistids.append([(len(allstrokeids) - 1)])
    else:
        segmentation_strokes.append([allstrokes[(len(allstrokeids) - 1)]])
        segmentation_strokeids.append([allstrokeids[(len(allstrokeids) - 1)]])
        segmentation_strokelistids.append([(len(allstrokeids) - 1)])


    return segmentation_strokes, segmentation_strokeids


def getsegmentation(listoftraces, listoftraceids, labels, filename):
    """
    Preprocesses the data and performs segmentation on the list
    of strokes
    :param listoftraces: list of traces
    :param listoftraceids: list of trace ids
    :param labels: list of labels
    :param filename: inkml filename
    :return: segmentation of strokes and their ids
    """
    for ind in range(len(listoftraces)):
        listoftraces[ind] = smoothdatapointsandinterpolate(removeconsecutiveduplicatepoints(listoftraces[ind]))
    allstrokes, allstrokeids = flattentraces(listoftraces, listoftraceids)
    minX, maxX, minY, maxY = getminmaxXYinexpression(allstrokes)
    mthresh = getmergingthreshold(minX, maxX, minY, maxY, len(allstrokes))
    segmentation_strokes, segmentation_strokeids = performnonbaselinesegmentation(allstrokes, allstrokeids, mthresh)
    # segmentation_strokes, segmentation_strokeids = \
    #     performbaselinesegmentation(segmentation_strokes, segmentation_strokeids)
    return segmentation_strokes, segmentation_strokeids


def separateXandYlistsinstrokes(segmentation_strokes):
    """
    Splits the X and Y coordinates and puts both of them
    in a common list one after another
    :param segmentation_strokes: segmentation strokes
    :return: X and Y separated list
    """
    separatedstrokes = []
    for outer_ind in range(len(segmentation_strokes)):
        persymbolstrokes = segmentation_strokes[outer_ind]
        separatedpersymbolstrokes = []
        for inner_ind in range(len(persymbolstrokes)):
            separatedpersymbolstrokes.append(persymbolstrokes[inner_ind][0])
            separatedpersymbolstrokes.append(persymbolstrokes[inner_ind][1])
        separatedstrokes.append(separatedpersymbolstrokes)

    return separatedstrokes


def normalizedatawithoutsmoothening(featureentries):
    """
    finds local min and max coordinates and rescales them between -1 and 1
    :param data: dataset features + labels
    :return: rescaled data
    """
    for ind_in_data in range(len(featureentries)):
        # if ind_in_data == 112111:
        #     print("error was here")
        featureentry = featureentries[ind_in_data]
        localminX = 999999.0
        localmaxX = -999999.0
        localminY = 999999.0
        localmaxY = -999999.0

        for ind_in_feature in range(int(len(featureentry)/2)):
            Xcoords = featureentry[2*ind_in_feature]
            Ycoords = featureentry[2*ind_in_feature + 1]

            if int(len(featureentry)/2) == 1 and len(Xcoords) == 1:
                localmaxY = Ycoords[0] + 20
                localminY = Ycoords[0] - 20
                localmaxX = Xcoords[0] + 10
                localminX = Xcoords[0] - 10
            else:
                # Find local mins and maxs
                for ind_in_coord in range(len(Xcoords)):
                    if Xcoords[ind_in_coord] < localminX:
                        localminX = Xcoords[ind_in_coord]
                    if Xcoords[ind_in_coord] > localmaxX:
                        localmaxX = Xcoords[ind_in_coord]
                    if Ycoords[ind_in_coord] < localminY:
                        localminY = Ycoords[ind_in_coord]
                    if Ycoords[ind_in_coord] > localmaxY:
                        localmaxY = Ycoords[ind_in_coord]

        # Rescale the min and max depending on which direction maximum variation is
        difference = abs((localmaxX - localminX) - (localmaxY - localminY)) / 2
        if (localmaxX - localminX) > (localmaxY - localminY):
            localmaxY = localmaxY + difference
            localminY = localminY - difference
        else:
            localmaxX = localmaxX + difference
            localminX = localminX - difference
        # print(ind_in_data)
        # print("Difference is " + str(difference) + " maxX " +str(localmaxX)+" minX " +str(localminX)+ " maxY "+str(localmaxY)+ " minY " +str(localminY))
        # featureentries[ind_in_data] = smoothdatapointsandinterpolate(removeconsecutiveduplicatepoints(featureentry))
        featureentries[ind_in_data] = scalealldatapoints(featureentries[ind_in_data], localmaxX, localminX, localmaxY, localminY)

    return featureentries


def performbulkclassification(model, bulk_featurevectors):
    """
    Given a list of list of features, this function uses a pretrained
    classifier to predict the symbols that these lists represent.
    :param model: trained classifier
    :param bulk_featurevectors: feature vectors
    :return: list of expressions. expressions contains symbols
    """
    all_expressions = []
    for featurevectors in bulk_featurevectors:
        symbols_in_expression = gettoptenoutputsfromrforest(model, featurevectors)
        for ind in range(len(symbols_in_expression)):
            symbols_in_expression[ind] = symbols_in_expression[ind][0]

        all_expressions.append(symbols_in_expression)

    return all_expressions


def producebulktestfeaturevectoranddosegmentation(data_test_traces_packed,
                                                  data_test_traceids_packed,
                                                  data_test_labels_packed ,
                                                  data_test_filenames_packed):
    """
    given a list of strokes, this function does segmentation of these
    strokes, extracts features and returns them
    :param data_test_traces_packed: list of traces
    :param data_test_traceids_packed: list of trace ids
    :param data_test_labels_packed: list of labels
    :param data_test_filenames_packed: list of filenames
    :return: strokes, stroke ids and feature vectors
    """
    bulk_segmentation_strokes = []
    bulk_segmentation_strokeids = []
    bulk_featurevectors = []
    for ind in range(len(data_test_filenames_packed)):
        segmentation_strokes, segmentation_strokeids = getsegmentation(data_test_traces_packed[ind],
                                                                       data_test_traceids_packed[ind],
                                                                       data_test_labels_packed[ind],
                                                                       data_test_filenames_packed[ind])
        separated_seg_strokes = separateXandYlistsinstrokes(segmentation_strokes)
        normalized_separated_seg_strokes = normalizedatawithoutsmoothening(separated_seg_strokes)
        feature_vector_test = extractfeaturesandcreatevector(normalized_separated_seg_strokes)
        bulk_featurevectors.append(feature_vector_test)
        bulk_segmentation_strokes.append(segmentation_strokes)
        bulk_segmentation_strokeids.append(segmentation_strokeids)
    return bulk_segmentation_strokes, bulk_segmentation_strokeids, bulk_featurevectors


def packdataperfileintooneentry(listoftraces, listoftraceids, labels, filenames):
    """
    Puts traces belonging to the same file in one list.
    :param listoftraces: list of traces
    :param listoftraceids: list of trace ids
    :param labels: list of labels
    :param filenames: list of filenames
    :return: trace data per file.
    """
    prevfilename = ""
    files = []
    labelsperfile = []
    tracesperfile = []
    traceidsperfile = []
    intertraces = []
    intertraceids = []
    interlabels = []
    for ind in range(len(listoftraces)):
        if filenames[ind] != prevfilename:
            if prevfilename != "":
                files.append(prevfilename)
                tracesperfile.append(intertraces)
                traceidsperfile.append(intertraceids)
                labelsperfile.append(interlabels)
            intertraces = []
            intertraceids = []
            interlabels = []
            prevfilename = filenames[ind]
        intertraces.append(listoftraces[ind])
        intertraceids.append(listoftraceids[ind])
        interlabels.append(labels[ind])

    # For the last file
    files.append(prevfilename)
    tracesperfile.append(intertraces)
    traceidsperfile.append(intertraceids)
    labelsperfile.append(interlabels)

    return tracesperfile, traceidsperfile, labelsperfile, files


def getcountofdistinctlabelsinallfiles(labels):
    """
    count the number of specific symbols in the dataset
    :param data_label: labels of the data
    :return: dictionary with count of the data labels
    """
    labelsandcounts = {}
    for onefilelabels in labels:
        for item in onefilelabels:
            if item in labelsandcounts:
                labelsandcounts[item] = labelsandcounts[item]+1
            else:
                labelsandcounts[item] = 1
    return labelsandcounts


def unpackdataperlabelintooneentry(listoftraces, listoftraceids, labels, filenames):
    """
    splits data per file into multiple entries
    :param listoftraces: list of traces
    :param listoftraceids: list of trace ids
    :param labels: list of labels
    :param filenames: list of files
    :return: split data
    """
    listoftraces_new = []
    listoftraceids_new = []
    labels_new = []
    filenames_new = []
    for ind in range(len(filenames)):
        for lblsindex in range(len(labels[ind])):
            listoftraces_new.append(listoftraces[ind][lblsindex])
            listoftraceids_new.append(listoftraceids[ind][lblsindex])
            labels_new.append(labels[ind][lblsindex])
            filenames_new.append(filenames[ind])

    return [listoftraces_new, listoftraceids_new, labels_new, filenames_new]


def splitfilestotraintest(listoftraces, listoftraceids, labels, filenames, labelsandcounts):
    """
    Creates a 2/3 and 1/3 Train and test split on the dataset
    :param listoftraces: list of traces
    :param listoftraceids: list of trace ids
    :param labels: list of labels
    :param filenames: list of files
    :param labelsandcounts: labels and their counts in the set
    :return: training data and testing data
    """
    traindatafeatures = []
    testdatafeatures = []
    traintraceids = []
    testtraceids = []
    traindatalabels = []
    testdatalabels = []
    trainfilenames = []
    testfilenames = []
    labelsandcurrentcounts = labelsandcounts.copy()
    # Set their current count to 0
    # Maintains their count as we split in train and test
    for item in labelsandcurrentcounts:
        labelsandcurrentcounts[item] = 0

    for ind in range(len(filenames)):
        includetotrain = False
        for lbls in labels[ind]:
            if labelsandcurrentcounts[lbls] < 0.66 * labelsandcounts[lbls]:
                includetotrain = True
                break
        if includetotrain:
            for lblsindex in range(len(labels[ind])):
                labelsandcurrentcounts[labels[ind][lblsindex]] = labelsandcurrentcounts[labels[ind][lblsindex]] + 1
                traindatafeatures.append(listoftraces[ind][lblsindex])
                traintraceids.append(listoftraceids[ind][lblsindex])
                traindatalabels.append(labels[ind][lblsindex])
                trainfilenames.append(filenames[ind])
        else:
            for lblsindex in range(len(labels[ind])):
                testdatafeatures.append(listoftraces[ind][lblsindex])
                testtraceids.append(listoftraceids[ind][lblsindex])
                testdatalabels.append(labels[ind][lblsindex])
                testfilenames.append(filenames[ind])

    return [traindatafeatures, traintraceids, traindatalabels, trainfilenames], \
           [testdatafeatures, testtraceids, testdatalabels, testfilenames]


def getjustfilenamewithoutpathorextension(fullfilename):
    """
    Returns a filename without directory or extension
    :param fullfilename: filename with directory and extn
    :return: filename without directory or extension
    """
    name = ""
    dotindex = fullfilename.rfind(".")
    lastbackslashindex = fullfilename.rfind("\\")
    return fullfilename[lastbackslashindex + 1:dotindex]


def producebulkLGfiles(data_test_filenames_packed, all_expressions, bulk_segmentation_strokeids, lgfilessavedirectory):
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
    for file_name in data_test_filenames_packed:
        orig_stdout = sys.stdout
        # output_name = os.path.splitext(file_name)[0]

        # creates a new file using the .inkml filename, with the .lg extension
        filename = getjustfilenamewithoutpathorextension(file_name)
        f = open(lgfilessavedirectory + "\\" + filename + ".lg", "w")
        sys.stdout = f
        # for i in range(len(trace_id)):
        # print("O,",trace_id,",",symbol,",",wt,",",strokes_no)
        for j in range(len(all_expressions[i])):
            sym = str(all_expressions[i][j])
            if sym == ',':
                sym = "COMMA"
            print("O, " + "obj_id_"+ str(j) + ", " + str(sym) + ", " + str(1.0) + ", " + str(
                bulk_segmentation_strokeids[i][j]).strip('[]'))
        i = i + 1
        sys.stdout = orig_stdout
    f.close()
    return None


def createdirectoryofinkmlfiles(filenames, directoryname):
    """
    Create directory and save files in it
    :param filenames: list of filenames
    :param directoryname: directory for save
    :return: None
    """
    if not os.path.exists(directoryname):
        os.makedirs(directoryname)
    for fn in filenames:
        shutil.copy(fn, directoryname)


def gettrainedrandomforestmodel(featurevector_train, data_train_labels):
    """
    Trains a Rforest model and returns it
    :param featurevector_train: feature vector train
    :param data_train_labels: train data labels
    :return: RForest model
    """
    # Train a model for random forest and create its pickle
    if os.path.isfile(FILENAME):
        return joblib.load(FILENAME)
    else:
        random_forest = RandomForestClassifier(bootstrap=True,
                                                class_weight=None, criterion='gini')
        random_forest.fit(featurevector_train, data_train_labels)
        filename = FILENAME

        with open(filename, 'wb') as f:
            joblib.dump(random_forest, f)
        return random_forest


def gettraceinfo(root, featureentry, traceidlist, traceid):
    """
    Reads the trace information in inkml files and save them
    in lists
    :param root: root tag in inkml file
    :param featureentry: trace points
    :param traceidlist: trace ids
    :param traceid: trace id
    :return: trace points and list of trace ids
    """
    Xcoords = []
    Ycoords = []

    for subel in root:
        if subel.tag == '{http://www.w3.org/2003/InkML}trace' and subel.items()[0][1] == traceid:
            tracestr = (subel.text).replace('\n','')
            coords = tracestr.split(',')
            for index in range(len(coords)):
                parts = coords[index].strip().split(' ')
                Xcoords.append(float(parts[0].strip()))
                Ycoords.append(float(parts[1].strip()))

    traceidlist.append(int(traceid))
    featureentry.append(Xcoords)
    featureentry.append(Ycoords)
    return featureentry, traceidlist


def readinkmlfiles(fn, listoftraces, listoftraceids, labels, filenames):
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
                        if subsubsubel.tag == '{http://www.w3.org/2003/InkML}traceView':
                            featureentry, traceidlist = gettraceinfo(root, featureentry, traceidlist, subsubsubel.items()[0][1])
                    onefilelistoftraces.append(featureentry)
                    onefilelistoftraceids.append(traceidlist)
    if listoftraces != None:
        listoftraces.append(onefilelistoftraces)
        listoftraceids.append(onefilelistoftraceids)
        labels.append(onefilelabels)
        filenames.append(fn)
        return listoftraces, listoftraceids, labels, filenames
    else:
        return onefilelistoftraces, onefilelistoftraceids, onefilelabels, fn


def readdirectoryforinkmlfiles(subdir, listoftraces, listoftraceids, labels, filenames):
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
            listoftraces, listoftraceids, labels, filenames = readinkmlfiles(subdir + "\\" +item, listoftraces, listoftraceids, labels, filenames)
    return listoftraces, listoftraceids, labels, filenames


def fetchandreaddataset(dir):
    """
    Reads dataset
    :param dir: TrainINKML directory
    :return: listoftraces, listoftraceids, labels, filenames
    """
    listoftraces = []
    listoftraceids = []
    labels = []
    filenames = []
    listdir = [x[0] for x in os.walk(dir)]
    for index in range(1,len(listdir)):
        listoftraces, listoftraceids, labels, filenames = readdirectoryforinkmlfiles(listdir[index], listoftraces, listoftraceids, labels, filenames)
    return listoftraces, listoftraceids, labels, filenames


def shufflemultiplelists(*ls):
    """
    zips multiple lists and shuffles them
    :param ls: bunch of lists
    :return: same shuffled lists
    """
    lst = list(zip(*ls))
    shuffle(lst)
    return zip(*lst)


def segmentaninkmlfile(filepath):
    """
    Given an INKML file, this function performs segmentation and predicts the
    expression.
    :param filepath: path of the file
    :return: None
    """
    # print(filepath)
    listoftraces, listoftraceids, labels, filenames = readinkmlfiles(filepath, [], [], [], [])
    print("Preprocessing the data.")
    recognizer = gettrainedrandomforestmodel(None, None)

    print("Performing Segmentation and creating feature vectors of the segments.")
    # data_test_traces_packed, data_test_traceids_packed, data_test_labels_packed , data_test_filenames_packed = \
    #     packdataperfileintooneentry(listoftraces, listoftraceids, labels, filenames)
    bulk_segmentation_strokes, bulk_segmentation_strokeids, bulk_featurevectors = \
        producebulktestfeaturevectoranddosegmentation(listoftraces, listoftraceids, labels, filenames)

    print("Running Recognizer (Classifier) on the created segments.")
    all_expressions = performbulkclassification(recognizer, bulk_featurevectors)

    print("saving LG file of the INKML file in the same directory")
    producebulkLGfiles(filenames, all_expressions, bulk_segmentation_strokeids, filepath[:filepath.rfind('\\')])
    print("LG file generated")


def doonetimetrainstuff():
    """
    Things that needs to be done just once
    Saves a trained random forest model as a pickle
    """
    maindatapath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project2 - Segmentation\RealProject\TrainINKML\TrainINKML"
    testtestinkmlpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project2 - Segmentation\RealProject\TestTestINKML"
    testtraininkmlpath = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project2 - Segmentation\RealProject\TestTrainINKML"
    lgfilessavedirectory = "D:\Rochester Institute of Technology\Semester 4\Pattern Recognition\Projects\Project2 - Segmentation\RealProject\TestTrainLG_base"

    #PART-1 = COMMENT THIS IF YOUR SPLIT IS READY
    # print("Reading all data.")
    # listoftraces, listoftraceids, labels, filenames = fetchandreaddataset(maindatapath)
    #
    # print("Shuffling the read inkml files")
    # listoftraces, listoftraceids, labels, filenames = shufflemultiplelists(listoftraces, listoftraceids, labels, filenames)
    #
    # print("Splitting the data into train and test roughly 2/3 and 1/3.")
    # labelsandcounts = getcountofdistinctlabelsinallfiles(labels)
    # data_train, data_test = splitfilestotraintest(listoftraces, listoftraceids, labels, filenames, labelsandcounts)
    # labelsandcounts_train = getcountofdistinctlabels(data_train[2])
    # labelsandcounts_test = getcountofdistinctlabels(data_test[2])
    # distinctfiles_train = set(data_train[3])
    # distinctfiles_test = set(data_test[3])
    # createdirectoryofinkmlfiles(distinctfiles_test, testtestinkmlpath)
    # createdirectoryofinkmlfiles(distinctfiles_train, testtraininkmlpath)
    #PART-1 = END

    #PART-2 = COMMENT THIS IS YOUR SPLIT IS NOT READY
    print("Reading split data.")
    listoftraces_train, listoftraceids_train, labels_train, filenames_train = readdirectoryforinkmlfiles(testtraininkmlpath, [], [], [], [])
    listoftraces_test, listoftraceids_test, labels_test, filenames_test = readdirectoryforinkmlfiles(testtestinkmlpath, [], [], [], [])
    data_train = unpackdataperlabelintooneentry(listoftraces_train, listoftraceids_train, labels_train, filenames_train)
    data_test = unpackdataperlabelintooneentry(listoftraces_test, listoftraceids_test, labels_test, filenames_test)
    #PART-2

    print("Preprocessing the data.")
    data_train_traces = normalizedata(data_train[0])
    data_train_traceids = data_train[1]
    data_train_labels = data_train[2]
    data_train_filenames = data_train[3]
    data_test_traces = data_test[0]
    data_test_traceids = data_test[1]
    data_test_labels = data_test[2]
    data_test_filenames = data_test[3]

    print("Extracting features, creating vector, and training a RF model.")
    # featurevector_train = extractfeaturesandcreatevector(data_train_traces)       # Keep this commented to save time
    # recognizer = gettrainedrandomforestmodel(featurevector_train, data_train_labels) # Keep this commented to save time
    # print("A trained Random Forest model is saved and ready for further use now!")
    recognizer = gettrainedrandomforestmodel(None, None)

    print("Performing Segmentation and creating feature vectors of the segments.")
    data_test_traces_packed, data_test_traceids_packed, data_test_labels_packed , data_test_filenames_packed = \
        packdataperfileintooneentry(data_test_traces, data_test_traceids, data_test_labels, data_test_filenames)
    bulk_segmentation_strokes, bulk_segmentation_strokeids, bulk_featurevectors = \
        producebulktestfeaturevectoranddosegmentation(data_test_traces_packed, data_test_traceids_packed, data_test_labels_packed, data_test_filenames_packed)

    print("Running Recognizer (Classifier) on the created segments.")
    all_expressions = performbulkclassification(recognizer, bulk_featurevectors)

    print("Recognition completed. Generating output lg files.")
    producebulkLGfiles(data_test_filenames_packed, all_expressions, bulk_segmentation_strokeids, lgfilessavedirectory)
    print("BULK SEGMENTATION - THE END!")


def main():
    """
    main method
    """
    # doonetimetrainstuff()
    print("STARTED SEGMENTATION")
    segmentaninkmlfile(sys.argv[1])
    print("SEGMENTATION - THE END!")


if __name__ == "__main__":
    main()