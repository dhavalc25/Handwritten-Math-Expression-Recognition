"""
project1.py

Authors: Dhaval Chauhan dmc8686@g.rit.edu
Authors: Bhavin Bhuta bsb5375@g.rit.edu

Preprocess CROHME dataset, extract features and do
classification of symbols using kdtree and random forest
"""


from bs4 import BeautifulSoup as Soup
from pylab import *
import numpy as np
from scipy import interpolate
import os
import csv
import _pickle as cPickle
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


LARGEDATASET_VALID = "\\task2-trainSymb2014\\trainingSymbols"
LARGEDATASET_JUNK = "\\task2-trainSymb2014\\trainingJunk"
BONUSTESTSET = "\\testSymbols\\testSymbols"


def readinkmlfiles(fn):
    """
    takes in an inkml filename and returns trace and id info
    :param fn: filename with extension .inkml
    :return: trace information and ids
    """
    id = ""
    inkmldatahandler = open(fn).read()
    inkmldata = Soup(inkmldatahandler, "html.parser")
    annotags = inkmldata.find_all('annotation')
    annotagUIstring = str(annotags[1])
    id = int(annotagUIstring[annotagUIstring.rfind("_")+1:annotagUIstring.index("</")])
    tracetags = inkmldata.find_all('trace')
    return tracetags, id


def readinkmlfileswithoutlabels(fn):
    """
    takes in an inkml filename and returns trace and id info
    :param fn: filename with extension .inkml
    :return: trace information and ids
    """
    id = ""
    inkmldatahandler = open(fn).read()
    inkmldata = Soup(inkmldatahandler, "html.parser")
    annotags = inkmldata.find_all('annotation')
    annotagUIstring = str(annotags[1])
    id = (annotagUIstring[annotagUIstring.find("\">")+2:annotagUIstring.index("</")])
    tracetags = inkmldata.find_all('trace')
    return tracetags, id


def gettracecoordinates(tracetags):
    """
    Takes in a list of tracetags and extracts trace coordinate information
    :param tracetags: list of traces
    :return: list of traces in form of floating coordinates
    """
    n = len(tracetags)
    listoftraces = [None]*n*2
    for ind in range(n):
        tracetagstring = str(tracetags[ind])

        # Get string of coordinates
        coordinatesstring = tracetagstring[tracetagstring.index("\">\n")+3:tracetagstring.index("\n</")]
        coordinatearray = coordinatesstring.split(",")
        listofXcoords = [0]*len(coordinatearray)
        listofYcoords = [0]*len(coordinatearray)
        for i in range(len(coordinatearray)):
            coord = coordinatearray[i].split()

            # Save each coordinate value as float
            x = float(coord[0].strip())
            y = float(coord[1].strip())
            listofXcoords[i] = x
            listofYcoords[i] = y

        # Save as list of C coordinates and list of Y coordinates
        # For each trace there will be 2 lists, 1 for Xs and 1 for Ys
        listoftraces[2*ind] = listofXcoords
        listoftraces[2*ind + 1] = listofYcoords

    # Return the coordinate lists
    return listoftraces


def getlabelbyid(gtdata,id):
    """
    Given an ID, this function fetches its label from the
    ground truth data obtained from iso-GT.txt file
    :param gtdata: Ground truth data
    :param id: unique ID of the data eg 101_alfonso_07
    :return: ground truth label ID of the data and the entire string itself
    """
    gtstring = str(gtdata[id])
    lbl = gtstring.split(",")[1]
    lbl = lbl.replace("\n","")
    return lbl, gtstring


def readgroundtruthfiles(csvfilename):
    """
    Reads the GT files
    :param csvfilename: filename without .csv
    :return: raw features, GT labels, Annotation string
    """
    # Get current working directory
    location = os.getcwd()

    if "iso" in csvfilename:
        # Go to the training symbols folder in the current directory
        location = location + LARGEDATASET_VALID
        filetypestring = "iso"

    else:
        # Go to the training symbols junk folder in the current directory
        location = location + LARGEDATASET_JUNK
        filetypestring = "junk"

    # Open the ground truth junk file and read all the data
    with open(csvfilename + ".csv") as gtfile:
        gtdata = csv.reader(gtfile, delimiter=',')
        data_features = []
        data_label = []
        data_gtstring = []
        for row in gtdata:
            # Add label to the labels list
            data_label.append(str(row[1]))
            gtstring = str(row[0])
            data_gtstring.append(gtstring.strip())
            id = (gtstring[gtstring.rfind("_")+1:])
            tracetags, id = readinkmlfiles(location + "\\" + filetypestring + id + ".inkml" )

            # Add trace/coordinate information to the raw features list
            data_features.append(gettracecoordinates(tracetags))

    # Return raw features and their labels
    return data_features, data_label, data_gtstring


def readnonCSVdatafiles(typeofdata):
    """
    Read the file and process the input data in a specific format
    :param typeofdata:valid or junk data
    :return: datafeatures and groundtruth labels of those features
    """

    # Get current working directory
    location = os.getcwd()
    gtfile = None

    if typeofdata == "Valid":
        # Go to the training symbols folder in the current directory
        location = location + LARGEDATASET_VALID

        # Open the ground truth file and read all the data
        gtfile = open(location + "\iso_GT.txt", "r")
    elif typeofdata == "Junk":
        # Go to the training symbols junk folder in the current directory
        location = location + LARGEDATASET_JUNK

        # Open the ground truth junk file and read all the data
        gtfile = open(location + "\junk_GT.txt", "r")
    elif typeofdata == "Test":
        # Go to the training symbols junk folder in the current directory
        location = location + BONUSTESTSET

        data_features = []
        data_gtstring = []

        # For every .inkml file in the trainingsymbols folder
        # read it and save it's trace information with label
        for fn in os.listdir(location):
            try:
                if fn.endswith(".inkml"):
                    tracetags, id = readinkmlfileswithoutlabels(location + "\\" + fn)
                    coordinates = gettracecoordinates(tracetags)
                    data_features.append(coordinates)
                    data_gtstring.append(id)
            except Exception as e:
                raise e
                print("File not found!")
        return data_features, data_gtstring

    gtdata = gtfile.readlines()
    numberofdata = len(gtdata)
    data_features = [None]*numberofdata
    data_label = [""]*numberofdata
    data_gtstring = [""]*numberofdata
    gtfile.close()
    fileindex = 0

    # For every .inkml file in the trainingsymbols folder
    # read it and save it's trace information with label
    for fn in os.listdir(location):
        try:
            if fn.endswith(".inkml"):
                tracetags, id = readinkmlfiles(location+"\\" +fn)
                coordinates = gettracecoordinates(tracetags)
                lbl, gtstring = getlabelbyid(gtdata,id)
                data_features[fileindex] = coordinates
                data_label[fileindex] = lbl
                data_gtstring[fileindex] = gtstring.replace("\n","")
                fileindex += 1
        except Exception as e:
            raise e
            print("File not found!")

    return data_features,data_label,data_gtstring


def getcountofdistinctlabels(data_label):
    """
    count the number of specific symbols in the dataset
    :param data_label: labels of the data
    :return: dictionary with count of the data labels
    """
    labelsandcounts = {}
    for item in data_label:
        if item in labelsandcounts:
            labelsandcounts[item] = labelsandcounts[item]+1
        else:
            labelsandcounts[item] = 1
    return labelsandcounts


def splitdatatotrainandtest(data_features, data_label, data_gtstring, labelsandcounts):
    """
    Divide training data into 70-30 approximately train-test data
    :param data_features: features of the data
    :param data_label: labels of the data
    :param labelsandcounts: count of the symbols present in the set
    :return: traindata and testdata
    """
    traindatafeatures = []
    testdatafeatures = []
    traindatalabels = []
    testdatalabels = []
    traindataGTstrings = []
    testdataGTstrings = []

    labelsandcurrentcounts = labelsandcounts.copy()

    # Set their current count to 0
    # Maintains their count as we split in train and test
    for item in labelsandcurrentcounts:
        labelsandcurrentcounts[item] = 0

    for ind in range(len(data_features)):
        # If count is less than 70% of the total then add data to train data
        if labelsandcurrentcounts[data_label[ind]] < 0.7*labelsandcounts[data_label[ind]]:
            labelsandcurrentcounts[data_label[ind]] = labelsandcurrentcounts[data_label[ind]] + 1
            traindatafeatures.append(data_features[ind])
            traindatalabels.append(data_label[ind])
            traindataGTstrings.append(data_gtstring[ind])
        # If count goes greater than 70% of the total then add it to test data
        else:
            testdatafeatures.append(data_features[ind])
            testdatalabels.append(data_label[ind])
            testdataGTstrings.append(data_gtstring[ind])

    #return both train and test data
    return [traindatafeatures, traindatalabels, traindataGTstrings],[testdatafeatures, testdatalabels, testdataGTstrings]


def interpolatesplinepoints(xcoords, ycoords, deg):
    """
    fits input points to a spline equation and get coefficients.
    generate 40 interpolated points and return them
    :param xcoords: list of x coordinates
    :param ycoords: list of y coordinates
    :param deg: highest degree of curve 1,2, or 3
    :return: interpolated points
    """
    tupletck, arrayu = interpolate.splprep([xcoords,ycoords], s=5.0,k=deg)
    noofinterpolationpoints = np.arange(0, 1.00, 0.025)
    out = interpolate.splev(noofinterpolationpoints, tupletck)
    return out[0], out[1]


def smoothdatapointsandinterpolate(featureentry):
    """
    Reads the trace information and sends the coordinate
    points to get their smooth interpolation points
    :param featureentry: list of traces
    :return: smooth traces
    """
    for outerind in range(int(len(featureentry)/2)):
        Xcoords = featureentry[2*outerind]
        Ycoords = featureentry[2*outerind + 1]
        if(len(Xcoords) == 2):
            featureentry[2 * outerind], featureentry[2*outerind + 1] = \
                interpolatesplinepoints(featureentry[2 * outerind], featureentry[2*outerind + 1], deg=1)
        if(len(Xcoords) == 3):
            featureentry[2 * outerind], featureentry[2*outerind + 1] = \
                interpolatesplinepoints(featureentry[2 * outerind], featureentry[2*outerind + 1], deg=2)
        if(len(Xcoords) > 3):
            featureentry[2 * outerind], featureentry[2*outerind + 1] = \
                interpolatesplinepoints(featureentry[2 * outerind], featureentry[2*outerind + 1], deg=3)

    return featureentry


def removeconsecutiveduplicatepoints(featureentry):
    """
    This function reads trace coordinate points and removes
    the consecutive duplicate points.
    :param featureentry: list of trace information
    :return: trace information without duplicate points
    """
    for outerind in range(int(len(featureentry)/2)):
        Xcoords = featureentry[2*outerind]
        Ycoords = featureentry[2*outerind + 1]
        prevX = Xcoords[0]
        prevY = Ycoords[0]
        xnew = [prevX]
        ynew = [prevY]
        for innerind in range(1, len(Xcoords)):
            x1 = Xcoords[innerind]
            y1 = Ycoords[innerind]
            if prevX != x1 or prevY != y1:
                xnew.append(x1)
                ynew.append(y1)
            prevX = x1
            prevY = y1
        featureentry[2 * outerind] = xnew
        featureentry[2 * outerind + 1] = ynew
    return featureentry


def scalealldatapoints(featureentry,maxX,minX,maxY,minY):
    """
    Given all trace information of a given symbol, and the mins
    and maxes of the coordinates, scale them between -1 and 1
    :param featureentry: trace information
    :param maxX: Max value of X coordinate in the symbol
    :param minX: Min value of X coordinate in the symbol
    :param maxY: Max value of Y coordinate in the symbol
    :param minY: Min value of Y coordinate in the symbol
    :return: data scaled between -1 and 1
    """
    scaleddata = featureentry.copy()

    for ind_in_feature in range(int(len(featureentry)/2)):
        Xcoords = scaleddata[2*ind_in_feature]
        Ycoords = scaleddata[2*ind_in_feature + 1]
        for ind_in_coord in range(len(Xcoords)):
            if (maxX - minX) == 0:
                maxX = Xcoords[ind_in_coord] + 10
                minX = Xcoords[ind_in_coord] - 10
            if (maxY - minY) == 0:
                maxY = Ycoords[ind_in_coord] + 20
                minY = Ycoords[ind_in_coord] - 20

            # Scale the values between -1 and 1
            # print(str(ind_in_feature)+" MaxX is :" + str(maxX) + " MinX is "+str(minX))
            Xcoords[ind_in_coord] = ((2*(Xcoords[ind_in_coord] - minX))/(maxX - minX)) - 1
            Ycoords[ind_in_coord] = ((2*(Ycoords[ind_in_coord] - minY))/(maxY - minY)) - 1

    # Return scaled data
    return scaleddata


def normalizedata(featureentries):
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
        featureentries[ind_in_data] = smoothdatapointsandinterpolate(removeconsecutiveduplicatepoints(featureentry))
        featureentries[ind_in_data] = scalealldatapoints(featureentries[ind_in_data], localmaxX, localminX, localmaxY, localminY)

    return featureentries


def getmeansofXandY(raw_feature):
    """
    Calculates means of X and Y coordinates
    :param raw_feature: list of traces in a symbol
    :return: meanX, meanY
    """
    sumX = 0
    sumY = 0
    count = 0
    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        for innerind in range(len(Xcoords)):
            count += 1
            sumX += Xcoords[innerind]
            sumY += Ycoords[innerind]
    return (sumX/count),(sumY/count)


def getcovariancebetweenXandY(raw_feature,meanX,meanY):
    """
    Calculates covariance between X and Y
    :param raw_feature: list of traces in a symbol
    :param meanX: mean of X
    :param meanY: mean of Y
    :return: CovXY
    """
    intercov = 0.0
    count = 0
    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        for innerind in range(len(Xcoords)):
            count += 1
            intercov += (Xcoords[innerind] - meanX)*(Ycoords[innerind] - meanY)
    return (intercov/(count))


def getlinelength(raw_feature):
    """
    Calculates the length of the lines that make up the symbol
    using euclidean distance formula.
    :param raw_feature: list of traces in a symbol
    :return: line length
    """
    linelength = 0.0
    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        for innerind in range(len(Xcoords) - 1):
            linelength += ((Xcoords[innerind] - Xcoords[innerind + 1])**2 +
                           (Ycoords[innerind] - Ycoords[innerind + 1])**2)**(1/2.0)
    return linelength


def checkifthesetwolinesintersect(x1,x2,x3,x4,y1,y2,y3,y4):
    """
    checks if two segments intersect at a point
    :param x1: x coordinate of point 1 of line 1
    :param x2: x coordinate of point 2 of line 1
    :param x3: x coordinate of point 1 of line 2
    :param x4: x coordinate of point 2 of line 2
    :param y1: y coordinate of point 1 of line 1
    :param y2: y coordinate of point 2 of line 1
    :param y3: y coordinate of point 1 of line 2
    :param y4: y coordinate of point 2 of line 2
    :return: True of false for intersection
    """
    if (((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) == 0):
        return False

    ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    if x3 > x4:
        x3,x4 = x4,x3

    if x1 <= ix and x2 >= ix and x3 <= ix and x4 >= ix:
        return True
    return False


def computecrossingfeatures(raw_feature,hormin,hormax,vermin,vermax):
    """
    Computes crossing feature for one given region
    :param raw_feature: list of trace information
    :param hormin: min horizontally
    :param hormax: max horizontally
    :param vermin: min vertically
    :param vermax: max vertically
    :return: average number of intersections
    """
    count = 0
    hordiff = abs(hormax - hormin)
    verdiff = abs(vermax - vermin)
    horstep = 0
    verstep = 0
    horitr = False
    if hordiff < verdiff:
        horitr = True
        horstep = hordiff/10.0
    else:
        verstep = verdiff/10.0

    for ind in range(10):
        if horitr:
            x1 = vermin
            x2 = vermax
            y1 = hormin + ind*horstep
            y2 = y1
        else:
            x1 = vermin + ind*verstep
            x2 = x1
            y1 = hormin
            y2 = hormax
        for outerind in range(int(len(raw_feature) / 2)):
            Xcoords = raw_feature[2 * outerind]
            Ycoords = raw_feature[2 * outerind + 1]
            for innerind in range(len(Xcoords) - 1):
                x3 = Xcoords[innerind]
                x4 = Xcoords[innerind + 1]
                y3 = Ycoords[innerind]
                y4 = Ycoords[innerind + 1]
                if checkifthesetwolinesintersect(x1,x2,x3,x4,y1,y2,y3,y4):
                    count += 1

    return count/10.0


def calculatemembershipvaluesforangles(raw_feature, i, j):
    """
    Computes average angles at a given corner points
    :param raw_feature: list of trace information
    :param i: x of corner point
    :param j: y of corner point
    :return: average angle at a corner point
    """
    x1 = -1 + i * 0.5
    y1 = -1 + j * 0.5
    count = 0
    membershipsum = 0

    for outerind in range(int(len(raw_feature) / 2)):
        Xcoords = raw_feature[2 * outerind]
        Ycoords = raw_feature[2 * outerind + 1]
        for innerind in range(len(Xcoords) - 1):
            x2 = Xcoords[innerind]
            y2 = Ycoords[innerind]
            x3 = Xcoords[innerind + 1]
            y3 = Ycoords[innerind + 1]

            # Get midpoint of the line
            xm = (x2+x3)/2.0
            ym = (y2+y3)/2.0

            count += 1

            # Calculate the actual euclidean distance between the two points
            actualeucdist = ((xm - x1) ** (2.0) + (ym - y1) ** (2.0)) ** (1 / 2.0)

            # Calculate the angle that the connected line makes with the horizontal
            thetarad = math.atan2(abs(ym - y1), abs(xm - x1))
            thetadeg = math.degrees(thetarad)

            if thetadeg > 45.0:
                permittedeucdist = 0.5 / math.sin(thetarad)
            elif thetadeg < 45.0:
                permittedeucdist = 0.5 / math.cos(thetarad)
            else:
                permittedeucdist = 0.5 / ((2) ** (1 / 2.0))

            if actualeucdist <= permittedeucdist:
                phirad = math.atan2((ym - y1), (xm - x1))
                phideg = math.degrees(phirad)

                if phideg < 180:
                    phideg = 180 + phideg

                membershipsum += phideg

    if count == 0:
        return 0
    else:
        return (membershipsum/count)


def computemembershipvaluesforpoints(raw_feature, i, j):
    """
    Computes average membership value at a corner point
    :param raw_feature: trice information
    :param i: x of corner point
    :param j: y of corner point
    :return: average membership value a point
    """
    x1 = -1 + i*0.5
    y1 = -1 + j*0.5
    count = 0
    membershipsum = 0

    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        for innerind in range(len(Xcoords)):
            x2 = Xcoords[innerind]
            y2 = Ycoords[innerind]
            count += 1

            # Calculate the actual euclidean distance between the two points
            actualeucdist = ((x2 - x1)**(2.0) + (y2 - y1)**(2.0))**(1/2.0)

            # Calculate the angle that the connected line makes with the horizontal
            thetarad = math.atan2(abs(y2-y1),abs(x2-x1))
            thetadeg = math.degrees(thetarad)

            if thetadeg > 45.0:
                permittedeucdist = 0.5/math.sin(thetarad)
            elif thetadeg < 45.0:
                permittedeucdist = 0.5/math.cos(thetarad)
            else:
                permittedeucdist = 0.5/((2)**(1/2.0))

            if actualeucdist <= permittedeucdist:
                membershipsum += ((0.5-abs(x2-x1))/(0.5))*((0.5-abs(y2-y1))/(0.5))

    return (membershipsum/count)


def calculateaspectratio(raw_feature):
    """
    Calculates aspect ratio of the symbol
    :param raw_feature: trace information
    :return: range horizontally, range vertically
    """
    localminX = 999999.0
    localmaxX = -999999.0
    localminY = 999999.0
    localmaxY = -999999.0
    for outerind in range(int(len(raw_feature)/2)):
        Xcoords = raw_feature[2*outerind]
        Ycoords = raw_feature[2*outerind + 1]
        for innerind in range(len(Xcoords) - 1):
            if Xcoords[innerind] < localminX:
                localminX = Xcoords[innerind]
            if Xcoords[innerind] > localmaxX:
                localmaxX = Xcoords[innerind]
            if Ycoords[innerind] < localminY:
                localminY = Ycoords[innerind]
            if Ycoords[innerind] > localmaxY:
                localmaxY = Ycoords[innerind]

    # Return horizontal reach, vertical reach
    return abs(localmaxX - localminX), abs(localmaxY - localminY)


def extractfeaturesandcreatevector(raw_features):
    """
    Extracts 42 different features and creates a vector of it
    :param raw_features: raw trace information
    :return: feature vector
    """
    numberoffeatures = 42
    feature_vector = np.zeros((numberoffeatures, len(raw_features)))

    for jj in range(len(raw_features)):
        # print(jj)
        # Calculate and add number of traces
        feature_vector[0][jj] = len(raw_features[jj])/2

        # Calculate and add mean of X and Y coordinates
        feature_vector[1][jj], feature_vector[2][jj] = getmeansofXandY(raw_features[jj])

        # Calculate covariance between X and Y
        feature_vector[3][jj] = getcovariancebetweenXandY(raw_features[jj], feature_vector[1][jj], feature_vector[2][jj])

        # Calculate overall line length of the symbol
        feature_vector[4][jj] = getlinelength(raw_features[jj])

        # Calculate aspect ratio  of the symbol
        feature_vector[5][jj], feature_vector[6][jj] = calculateaspectratio(raw_features[jj])

        # bottommost horizontal region crossing feature
        feature_vector[7][jj] = computecrossingfeatures(raw_features[jj],-1.0,-0.6,-1.0,1.0)

        # second horizontal region crossing feature
        feature_vector[8][jj] = computecrossingfeatures(raw_features[jj],-0.6,-0.2,-1.0,1.0)

        # third horizontal region crossing feature
        feature_vector[9][jj] = computecrossingfeatures(raw_features[jj],-0.2,0.2,-1.0,1.0)

        # fourth horizontal region crossing feature
        feature_vector[10][jj] = computecrossingfeatures(raw_features[jj],0.2,0.6,-1.0,1.0)

        # topmost horizontal region crossing feature
        feature_vector[11][jj] = computecrossingfeatures(raw_features[jj],0.6,1.0,-1.0,1.0)

        # leftmost vertical region crossing feature
        feature_vector[12][jj] = computecrossingfeatures(raw_features[jj],-1.0,1.0,-1.0,-0.6)

        # second vertical region crossing feature
        feature_vector[13][jj] = computecrossingfeatures(raw_features[jj],-1.0,1.0,-0.6,-0.2)

        # third vertical region crossing feature
        feature_vector[14][jj] = computecrossingfeatures(raw_features[jj],-1.0,1.0,-0.2,0.2)

        # fourth vertical region crossing feature
        feature_vector[15][jj] = computecrossingfeatures(raw_features[jj],-1.0,1.0,0.2,0.6)

        # rightmost vertical region crossing feature
        feature_vector[16][jj] = computecrossingfeatures(raw_features[jj],-1.0,1.0,0.6,1.0)

        # calculate fuzzy histogram of points (membership values at 25 corner points)
        for ii in range(25):
            ind_i = ii//5
            ind_j = ii - (ind_i*5)
            val = computemembershipvaluesforpoints(raw_features[jj], ind_i, ind_j)
            feature_vector[17 + ii][jj] = val

        # calculate fuzzy histogram of angles (membership values at 25 corner points)
        # for ii in range(25):
        #     ind_i = ii//5
        #     ind_j = ii - (ind_i*5)
        #     val = calculatemembershipvaluesforangles(raw_features[jj], ind_i, ind_j)
        #     feature_vector[42 + ii][jj] = val

    return feature_vector.transpose()


def createCSVfiles(gtstring, filename):
    """
    Saves contents as CSV files
    :param gtstring: content to be stored
    :param filename: name of the file
    :return: Nothing
    """
    np.savetxt(filename + ".csv", np.asarray(gtstring),fmt="%s")
    return None


def gettoptenoutputsfromrforest(trainedmodel, testfeaturevector):
    """
    Fetches top tep labels from random forest's predictions
    :param trainedmodel: object of trained model
    :param testfeaturevector: feature vector of test data
    :return: top ten labels for the feature vector
    """
    labels_final = []
    top_10 = 10
    X1 = trainedmodel.predict_proba(testfeaturevector)
    X1 = X1.tolist()
    label_list = trainedmodel.classes_

    for rows in range(len(X1)):
        max = -1
        max1 = []
        for top_k in range(top_10):
            for cols in range(len(X1[0])):
                # for elem in X1[rows]:
                if X1[rows][cols] > max:
                    col_to_remove = cols
                    max = X1[rows][cols]
            max1.append(label_list[col_to_remove])
            X1[rows][col_to_remove] = -1
            max = -1
        labels_final.append(max1)

    return labels_final


def gettoptenoutputsfromkdtree(trainedmodel, testfeaturevector, data_label):
    """
    Fetches top tep labels from random forest's predictions
    :param trainedmodel: object of trained model
    :param testfeaturevector: feature vector of test data
    :param trainedmodel:
    :param testfeaturevector:
    :param data_label: ground truth labels of train data
    :return: top ten labels for the feature vector
    """
    number_of_nearest = 100
    closest_100_final = []
    total_list = []

    X = trainedmodel.kneighbors(testfeaturevector, number_of_nearest)
    xy = X[1].shape

    for i in range(xy[0]):
        inter_list = X[1][i].tolist()
        closest_100_final.append(inter_list)

    for data_pts in range(xy[0]):
        for neighbors in range(xy[1]):
            index = closest_100_final[data_pts][neighbors]
            closest_100_final[data_pts][neighbors] = data_label[index]

    for i in range(xy[0]):
        top_10 = []
        [top_10.append(x) for x in closest_100_final[i] if x
         not in top_10]
        total_list.append(top_10[:10])

    final_list = np.empty(shape=(10,len(total_list)),dtype="<U6")
    for ind_i in range(len(total_list)):
        item = total_list[ind_i]
        for ind_j in range(len(item)):

            final_list[ind_j][ind_i] = str(item[ind_j])
    return final_list


def classifyusingKDtree(bonusornonbonus, junkornojunk, picklefilename):
    """
    Reads the necessary parameters and fits/predicts using KDTree model
    :param bonusornonbonus: for bonus or not for bonnus
    :param junkornojunk: include junk or don't include junk
    :param picklefilename: name of the trained pickle file
    :return:
    """

    # Check if bonus or nonbonus
    if bonusornonbonus == "bonus":
        if picklefilename == "retrain":
            filename = "kdtree_junk_bonus.sav"

            # Get features and labels of the train data
            print("Reading Train and test data files for bonus")
            data_features_valid, data_label_valid, data_gtstring_valid = readnonCSVdatafiles("Valid")
            data_features_junk, data_label_junk, data_gtstring_junk = readnonCSVdatafiles("Junk")
            data_features_test, data_gtstring_test = readnonCSVdatafiles("Test")

            # Merge junk data with valid data
            print("Merge Valid data with junk data for train data")
            data_features_valid.extend(data_features_junk)
            data_label_valid.extend(data_label_junk)
            data_gtstring_valid.extend(data_gtstring_junk)

            # Normalize the coordinate points in trace information between -1 and 1
            print("Normalizing the coordinate points in trace information to in between -1 and 1.")
            normalizeddatatrain = normalizedata(data_features_valid)
            normalizeddatatest = normalizedata(data_features_test)

            # Extract feature vector from raw data
            if os.path.isfile("bonus_train_features.csv"):
                print("Using stored feature vectors!")
                featurevectortrain = np.genfromtxt("bonus_train_features.csv", delimiter=',')
            else:
                print("Creating feature vectors!")
                featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                np.savetxt("bonus_train_features.csv", featurevectortrain, delimiter=", ")

            if os.path.isfile("bonus_test_features.csv"):
                featurevectortest = np.genfromtxt("bonus_test_features.csv", delimiter=',')
            else:
                featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                np.savetxt("bonus_test_features.csv", featurevectortest, delimiter=", ")

            # Train a model and create its pickle
            neigh = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
            neigh.fit(featurevectortrain, data_label_valid)
            # with open(filename, 'wb') as f:
            #     joblib.dump(neigh, f)
            #
            # # Test the test data with this model
            # # pickle_kd = open(filename, 'rb')
            # neigh = joblib.load(filename)

            toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid)
            # create output file
            outputfiledata = [data_gtstring_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                              toptenpredicts[3]
                , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                              toptenpredicts[9]]
            outputfiledata = np.asarray(outputfiledata).transpose()


            np.savetxt("kdtree-output-bonus.csv", np.asarray(outputfiledata), delimiter=", ",
                       fmt="%s")

            print("Prediction file kdtree-output-bonus.csv has been created.")
        else:

            # Get features and labels of the test data
            print("Reading Train and test data files for bonus")
            data_features_valid, data_label_valid, data_gtstring_valid = readnonCSVdatafiles("Valid")
            data_features_junk, data_label_junk, data_gtstring_junk = readnonCSVdatafiles("Junk")
            data_features_test, data_gtstring_test = readnonCSVdatafiles("Test")

            print("Merge Valid data with junk data for train data")
            data_features_valid.extend(data_features_junk)
            data_label_valid.extend(data_label_junk)
            data_gtstring_valid.extend(data_gtstring_junk)

            # Normalize the coordinate points in trace information between -1 and 1
            print("Normalizing the coordinate points in trace information to in between -1 and 1.")
            normalizeddatatest = normalizedata(data_features_test)

            if os.path.isfile("bonus_test_features.csv"):
                print("Using stored feature vectors!")
                featurevectortest = np.genfromtxt("bonus_test_features.csv", delimiter=',')
            else:
                print("Creating feature vectors!")
                featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                np.savetxt("bonus_test_features.csv", featurevectortest, delimiter=", ")

            # Test the test data with this model
            # pickle_kd = open(picklefilename, 'rb')
            neigh = joblib.load(picklefilename)

            toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid)
            # create output file
            outputfiledata = [data_gtstring_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                              toptenpredicts[3]
                , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                              toptenpredicts[9]]
            outputfiledata = np.asarray(outputfiledata).transpose()

            np.savetxt("kdtree-output-bonus.csv", np.asarray(outputfiledata), delimiter=", ",
                       fmt="%s")

            print("Prediction file kdtree-output-bonus.csv has been created.")

    if bonusornonbonus == "nonbonus":
        if junkornojunk == "junk":
            if picklefilename == "retrain":

                filename = "kdtree_junk_nonbonus.sav"

                # Get features and labels of the data from the .csv files
                print("Ok! reading the previously built .csv files and using them.")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")
                data_features_junk_train, data_label_junk_train, data_gtstring_junk_train = readgroundtruthfiles("junk_GT_train")
                data_features_junk_test, data_label_junk_test, data_gtstring_junk_test = readgroundtruthfiles("junk_GT_test")

                # Merge Junk into Valid data
                print("Merging junk data with valid data!")
                data_features_valid_train.extend(data_features_junk_train)
                data_label_valid_train.extend(data_label_junk_train)
                data_gtstring_valid_train.extend(data_gtstring_junk_train)
                data_features_valid_test.extend(data_features_junk_test)
                data_label_valid_test.extend(data_label_junk_test)
                data_gtstring_valid_test.extend(data_gtstring_junk_test)

                # Normalize the coordinate points in trace information between -1 and 1
                print("Normalizing the coordinate points in trace information to in between -1 and 1.")
                normalizeddatatrain = normalizedata(data_features_valid_train)
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_junk_train_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortrain = np.genfromtxt("iso_junk_train_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                    np.savetxt("iso_junk_train_features.csv", featurevectortrain,delimiter=", ")

                if os.path.isfile("iso_junk_test_features.csv"):
                    featurevectortest = np.genfromtxt("iso_junk_test_features.csv",delimiter=',')
                else:
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_junk_test_features.csv", featurevectortest,delimiter=", ")

                # Train a model and create its pickle
                neigh = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
                neigh.fit(featurevectortrain, data_label_valid_train)
                # with open(filename,'wb') as f:
                #     joblib.dump(neigh, f)
                #
                # # Test the test data with this model
                # # pickle_kd = open(filename,'rb')
                # neigh = joblib.load(filename)

                toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid_train)
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                                  toptenpredicts[3]
                    , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                                  toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()


                np.savetxt("kdtree-output-iso_junk_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")
                acc = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print("Classifying rate is "+str(acc)+"%")

            else:

                # Read test files
                print("Reading test files for testing")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")
                data_features_junk_train, data_label_junk_train, data_gtstring_junk_train = readgroundtruthfiles("junk_GT_train")
                data_features_junk_test, data_label_junk_test, data_gtstring_junk_test = readgroundtruthfiles("junk_GT_test")

                # Merge Junk into Valid data
                data_features_valid_train.extend(data_features_junk_train)
                data_label_valid_train.extend(data_label_junk_train)
                data_gtstring_valid_train.extend(data_gtstring_junk_train)
                data_features_valid_test.extend(data_features_junk_test)
                data_label_valid_test.extend(data_label_junk_test)
                data_gtstring_valid_test.extend(data_gtstring_junk_test)

                # Normalize the coordinate points in trace information between -1 and 1
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_junk_test_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortest = np.genfromtxt("iso_junk_test_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_junk_test_features.csv", featurevectortest,delimiter=", ")

                # Test the test data with this model
                # pickle_kd = open(picklefilename,'rb')
                neigh = joblib.load(picklefilename)


                toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid_train)
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                                  toptenpredicts[3]
                    , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                                  toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()

                np.savetxt("kdtree-output-iso_junk_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print("Classifying rate is "+str(acc)+"%")

        if junkornojunk == "nojunk":
            if picklefilename == "retrain":
                filename = "kdtree_nojunk_nonbonus.sav"

                # Get features and labels of the data from the .csv files
                print("Ok! reading the previously built .csv files and using them.")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")

                # Normalize the coordinate points in trace information between -1 and 1
                print("Normalizing the coordinate points in trace information to in between -1 and 1.")
                normalizeddatatrain = normalizedata(data_features_valid_train)
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_train_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortrain = np.genfromtxt("iso_train_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                    np.savetxt("iso_train_features.csv", featurevectortrain,delimiter=", ")

                if os.path.isfile("iso_test_features.csv"):
                    featurevectortest = np.genfromtxt("iso_test_features.csv",delimiter=',')
                else:
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_test_features.csv", featurevectortest,delimiter=", ")

                # Train a model and create its pickle
                neigh = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
                neigh.fit(featurevectortrain, data_label_valid_train)
                # with open(filename, 'wb') as f:
                #     joblib.dump(neigh, f)
                #
                # # Test the test data with this model
                # # pickle_kd = open(filename, 'rb')
                # neigh = joblib.load(filename)


                toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid_train)
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                                  toptenpredicts[3]
                    , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                                  toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()

                np.savetxt("kdtree-output-iso_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")
                acc = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print("Classifying rate is " + str(acc) + "%")

            else:

                # Read test files
                print("Reading test files for testing")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")

                # Normalize the coordinate points in trace information between -1 and 1
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_test_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortest = np.genfromtxt("iso_test_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_test_features.csv", featurevectortest,delimiter=", ")

                # Test the test data with this model
                # pickle_kd = open(picklefilename, 'rb')
                neigh = joblib.load(picklefilename)

                toptenpredicts = gettoptenoutputsfromkdtree(neigh, featurevectortest, data_label_valid_train)
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                                  toptenpredicts[3]
                    , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                                  toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()

                np.savetxt("kdtree-output-iso_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print("Classifying rate is " + str(acc) + "%")


def classifyusingrandomforest(bonusornonbonus, junkornojunk, picklefilename):
    """
    reads given parameters and fits/predicts on a random forest model
    max_depth set to nolimit in Random forest
    Number of trees is default 10
    :param bonusornonbonus: for bonus or not for bonus
    :param junkornojunk: include junk or not include junk
    :param picklefilename: name of the trained pickle file
    :return:
    """

    # Check if bonus or nonbonus
    if bonusornonbonus == "bonus":
        if picklefilename == "retrain":
            filename = "rforest_junk_bonus.sav"

            # Get features and labels of the train data
            print("Reading Train and test data files for bonus")
            data_features_valid, data_label_valid, data_gtstring_valid = readnonCSVdatafiles("Valid")
            data_features_junk, data_label_junk, data_gtstring_junk = readnonCSVdatafiles("Junk")
            data_features_test, data_gtstring_test = readnonCSVdatafiles("Test")

            # Merge junk data with valid data
            print("Merge Valid data with junk data for train data")
            data_features_valid.extend(data_features_junk)
            data_label_valid.extend(data_label_junk)
            data_gtstring_valid.extend(data_gtstring_junk)

            # Normalize the coordinate points in trace information between -1 and 1
            print("Normalizing the coordinate points in trace information to in between -1 and 1.")
            normalizeddatatrain = normalizedata(data_features_valid)
            normalizeddatatest = normalizedata(data_features_test)

            # Extract feature vector from raw data
            if os.path.isfile("bonus_train_features.csv"):
                print("Using stored feature vectors!")
                featurevectortrain = np.genfromtxt("bonus_train_features.csv", delimiter=',')
            else:
                print("Creating feature vectors!")
                featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                np.savetxt("bonus_train_features.csv", featurevectortrain, delimiter=", ")

            if os.path.isfile("bonus_test_features.csv"):
                featurevectortest = np.genfromtxt("bonus_test_features.csv", delimiter=',')
            else:
                featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                np.savetxt("bonus_test_features.csv", featurevectortest, delimiter=", ")

            # Train a model for random forest and create its pickle
            random_forrest = RandomForestClassifier( bootstrap=True,
                                                    class_weight=None, criterion='gini')
            random_forrest.fit(featurevectortrain, data_label_valid)
            # with open(filename, 'wb') as f:
            #     joblib.dump(random_forrest, f)
            #
            # # Open a trained random forest pickled model and test the test data
            # # pickle_rf = open(filename, 'rb')
            # random_forrest = joblib.load(filename)

            toptenpredicts = gettoptenoutputsfromrforest(random_forrest, featurevectortest)
            toptenpredicts = np.asarray(toptenpredicts).transpose()
            # create output file
            outputfiledata = [data_gtstring_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                              toptenpredicts[3]
                , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                              toptenpredicts[9]]
            outputfiledata = np.asarray(outputfiledata).transpose()

            np.savetxt("rforest-output-bonus.csv", np.asarray(outputfiledata), delimiter=", ",
                       fmt="%s")

            print("Prediction file rforest-output-bonus.csv has been created.")
        else:

            # Get features and labels of the test data
            print("Reading Train and test data files for bonus")
            data_features_test, data_gtstring_test = readnonCSVdatafiles("Test")

            # Normalize the coordinate points in trace information between -1 and 1
            print("Normalizing the coordinate points in trace information to in between -1 and 1.")
            normalizeddatatest = normalizedata(data_features_test)

            if os.path.isfile("bonus_test_features.csv"):
                print("Using stored feature vectors!")
                featurevectortest = np.genfromtxt("bonus_test_features.csv", delimiter=',')
            else:
                print("Creating feature vectors!")
                featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                np.savetxt("bonus_test_features.csv", featurevectortest, delimiter=", ")

            # Open a trained random forest pickled model and test the test data
            # pickle_rf = open(picklefilename, 'rb')
            random_forrest = joblib.load(picklefilename)

            toptenpredicts = gettoptenoutputsfromrforest(random_forrest, featurevectortest)
            toptenpredicts = np.asarray(toptenpredicts).transpose()
            # create output file
            outputfiledata = [data_gtstring_test, toptenpredicts[0], toptenpredicts[1], toptenpredicts[2],
                              toptenpredicts[3]
                , toptenpredicts[4], toptenpredicts[5], toptenpredicts[6], toptenpredicts[7], toptenpredicts[8],
                              toptenpredicts[9]]
            outputfiledata = np.asarray(outputfiledata).transpose()

            # create output file
            np.savetxt("rforest-output-bonus.csv", np.asarray(outputfiledata), delimiter=", ",
                       fmt="%s")

            print("Prediction file rforest-output-bonus.csv has been created.")


    if bonusornonbonus == "nonbonus":
        if junkornojunk == "junk":
            if picklefilename == "retrain":
                filename = "rforest_junk_nonbonus.sav"

                # Get features and labels of the data from the .csv files
                print("Ok! reading the previously built .csv files and using them.")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")
                data_features_junk_train, data_label_junk_train, data_gtstring_junk_train = readgroundtruthfiles("junk_GT_train")
                data_features_junk_test, data_label_junk_test, data_gtstring_junk_test = readgroundtruthfiles("junk_GT_test")

                # Merge Junk into Valid data
                print("Merging junk data with valid data!")
                data_features_valid_train.extend(data_features_junk_train)
                data_label_valid_train.extend(data_label_junk_train)
                data_gtstring_valid_train.extend(data_gtstring_junk_train)
                data_features_valid_test.extend(data_features_junk_test)
                data_label_valid_test.extend(data_label_junk_test)
                data_gtstring_valid_test.extend(data_gtstring_junk_test)

                # Normalize the coordinate points in trace information between -1 and 1
                print("Normalizing the coordinate points in trace information to in between -1 and 1.")
                normalizeddatatrain = normalizedata(data_features_valid_train)
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_junk_train_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortrain = np.genfromtxt("iso_junk_train_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                    np.savetxt("iso_junk_train_features.csv", featurevectortrain,delimiter=", ")

                if os.path.isfile("iso_junk_test_features.csv"):
                    featurevectortest = np.genfromtxt("iso_junk_test_features.csv",delimiter=',')
                else:
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_junk_test_features.csv", featurevectortest,delimiter=", ")

                # Train a model for random forest and create its pickle
                random_forrest = RandomForestClassifier(bootstrap=True,
                                                        class_weight=None, criterion='gini')
                random_forrest.fit(featurevectortrain, data_label_valid_train)
                # with open(filename, 'wb') as f:
                #     joblib.dump(random_forrest, f)
                #
                # # Open a trained random forest pickled model and test the test data
                # # pickle_rf = open(filename, 'rb')
                # random_forrest = joblib.load(filename)

                toptenpredicts = gettoptenoutputsfromrforest(random_forrest,featurevectortest)
                toptenpredicts = np.asarray(toptenpredicts).transpose()
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0],toptenpredicts[1],toptenpredicts[2],toptenpredicts[3]
                    , toptenpredicts[4],toptenpredicts[5],toptenpredicts[6],toptenpredicts[7],toptenpredicts[8],toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()
                np.savetxt("rforest-output-iso_junk_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc1 = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print(acc1)

            else:

                # Read test files
                print("Reading test files for testing")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")
                data_features_junk_test, data_label_junk_test, data_gtstring_junk_test = readgroundtruthfiles("junk_GT_test")

                # Merge Junk into Valid data
                data_features_valid_test.extend(data_features_junk_test)
                data_label_valid_test.extend(data_label_junk_test)
                data_gtstring_valid_test.extend(data_gtstring_junk_test)

                # Normalize the coordinate points in trace information between -1 and 1
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_junk_test_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortest = np.genfromtxt("iso_junk_test_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_junk_test_features.csv", featurevectortest,delimiter=", ")

                # Open a trained random forest pickled model and test the test data
                # pickle_rf = open(picklefilename, 'rb')
                random_forrest = joblib.load(picklefilename)

                toptenpredicts = gettoptenoutputsfromrforest(random_forrest,featurevectortest)
                toptenpredicts = np.asarray(toptenpredicts).transpose()
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0],toptenpredicts[1],toptenpredicts[2],toptenpredicts[3]
                    , toptenpredicts[4],toptenpredicts[5],toptenpredicts[6],toptenpredicts[7],toptenpredicts[8],toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()
                np.savetxt("rforest-output-iso_junk_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc1 = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print(acc1)

        if junkornojunk == "nojunk":
            if picklefilename == "retrain":
                filename = "rforest_nojunk_nonbonus.sav"

                # Get features and labels of the data from the .csv files
                print("Ok! reading the previously built .csv files and using them.")
                data_features_valid_train, data_label_valid_train, data_gtstring_valid_train = readgroundtruthfiles("iso_GT_train")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")

                # Normalize the coordinate points in trace information between -1 and 1
                print("Normalizing the coordinate points in trace information to in between -1 and 1.")
                normalizeddatatrain = normalizedata(data_features_valid_train)
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_train_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortrain = np.genfromtxt("iso_train_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortrain = extractfeaturesandcreatevector(normalizeddatatrain)
                    np.savetxt("iso_train_features.csv", featurevectortrain,delimiter=", ")

                if os.path.isfile("iso_test_features.csv"):
                    featurevectortest = np.genfromtxt("iso_test_features.csv",delimiter=',')
                else:
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_test_features.csv", featurevectortest,delimiter=", ")

                # Train a model for random forest and create its pickle
                random_forrest = RandomForestClassifier( bootstrap=True,
                                                        class_weight=None, criterion='gini')
                random_forrest.fit(featurevectortrain, data_label_valid_train)
                # with open(filename, 'wb') as f:
                #     joblib.dump(random_forrest, f)
                #
                # # Open a trained random forest pickled model and test the test data
                # # pickle_rf = open(filename, 'rb')
                # random_forrest = joblib.load(filename)

                toptenpredicts = gettoptenoutputsfromrforest(random_forrest,featurevectortest)
                toptenpredicts = np.asarray(toptenpredicts).transpose()
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0],toptenpredicts[1],toptenpredicts[2],toptenpredicts[3]
                    , toptenpredicts[4],toptenpredicts[5],toptenpredicts[6],toptenpredicts[7],toptenpredicts[8],toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()
                np.savetxt("rforest-output-iso_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc1 = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print(acc1)

            else:

                # Read test files
                print("Reading test files for testing")
                data_features_valid_test, data_label_valid_test, data_gtstring_valid_test = readgroundtruthfiles("iso_GT_test")

                # Normalize the coordinate points in trace information between -1 and 1
                normalizeddatatest = normalizedata(data_features_valid_test)

                # Extract feature vector from raw data
                if os.path.isfile("iso_test_features.csv"):
                    print("Using stored feature vectors!")
                    featurevectortest = np.genfromtxt("iso_test_features.csv",delimiter=',')
                else:
                    print("Creating feature vectors!")
                    featurevectortest = extractfeaturesandcreatevector(normalizeddatatest)
                    np.savetxt("iso_test_features.csv", featurevectortest,delimiter=", ")

                # Open a trained random forest pickled model and test the test data
                # pickle_rf = open(picklefilename, 'rb')
                random_forrest = joblib.load(picklefilename)

                toptenpredicts = gettoptenoutputsfromrforest(random_forrest,featurevectortest)
                toptenpredicts = np.asarray(toptenpredicts).transpose()
                # create output file
                outputfiledata = [data_gtstring_valid_test, toptenpredicts[0],toptenpredicts[1],toptenpredicts[2],toptenpredicts[3]
                    , toptenpredicts[4],toptenpredicts[5],toptenpredicts[6],toptenpredicts[7],toptenpredicts[8],toptenpredicts[9]]
                outputfiledata = np.asarray(outputfiledata).transpose()
                np.savetxt("rforest-output-iso_GT_test.csv", np.asarray(outputfiledata),delimiter=", ",fmt="%s")

                acc1 = accuracy_score(data_label_valid_test, toptenpredicts[0]) * 100
                print(acc1)


def main():
    """
    new main method
    """
    selval = 0
    while selval == 0:

        # Select from bonus and non bonus
        selval = input("Select a number and press enter.\n"
                       "1. Run Bonus part (Uses 100% of train data and real test data to test)\n"
                       "2. Run Non-Bonus part (Uses 70% train - 30% test split of training data)\n")

        # If Bonus
        if selval == "1" or selval == 1:
            clsval = 0
            while clsval == 0:

                # Select which classifier you want to use
                clsval = input("RUNNING FOR BONUS: Select a classifier and press enter.\n"
                               "1. KDtree\n"
                               "2. Random Forest\n")

                # For KDTree
                if clsval == "1" or clsval == 1:

                    # If a pickled model is found
                    if os.path.isfile("kdtree_junk_bonus.sav"):
                        ynbool = 0
                        while ynbool == 0:

                            # Use this pickle or retrain and get a new pickle to use
                            ynbool = input("A pickled model already exist for KDTree + Junk + Bonus.\n"
                                           "Would you like to use that or get a new trained model? Select and press enter\n"
                                           "y/Y. Use the previously pickled model\n"
                                           "n/N. Retrain the model, get a new pickle and use that\n")

                            # If you want to use previously pickled version
                            if ynbool == "y" or ynbool == "Y":
                                classifyusingKDtree("bonus", "", "kdtree_junk_bonus.sav")

                            # If you want to retrain a new model and pickle and use it
                            elif ynbool == "n" or ynbool == "N":
                                classifyusingKDtree("bonus", "", "retrain")

                            # Invalid selection
                            else:
                                print("Invalid selection. Retry!")
                                ynbool = 0

                    # If a pickled model is not found then train the model and create a pickle
                    else:
                        classifyusingKDtree("bonus", "", "retrain")


                # For Random Forest
                elif clsval == "2" or clsval == 2:

                    # If a pickled model is found
                    if os.path.isfile("rforest_junk_bonus.sav"):
                        ynbool = 0
                        while ynbool == 0:

                            # Use this pickle or retrain and get a new pickle to use
                            ynbool = input("A pickled model already exist for RForest + Junk + Bonus.\n"
                                           "Would you like to use that or get a new trained model? Select and press enter\n"
                                           "y/Y. Use the previously pickled model\n"
                                           "n/N. Retrain the model, get a new pickle and use that\n")

                            # If you want to use previously pickled version
                            if ynbool == "y" or ynbool == "Y":
                                classifyusingrandomforest("bonus", "", "rforest_junk_bonus.sav")

                            # If you want to retrain a new model and pickle and use it
                            elif ynbool == "n" or ynbool == "N":
                                classifyusingrandomforest("bonus", "", "retrain")

                            # Invalid selection
                            else:
                                print("Invalid selection. Retry!")
                                ynbool = 0

                    # If a pickled model is not found then train the model and create a pickle
                    else:
                        classifyusingrandomforest("bonus", "", "retrain")


                # Invalid selection
                else:
                    print("Invalid selection. Retry!")
                    clsval = 0


        # If Non Bonus
        elif selval == "2" or selval == 2:
            clsval = 0
            while clsval == 0:

                # Select which classifier you want to use
                clsval = input("RUNNING FOR NON-BONUS: Select a classifier and press enter.\n"
                               "1. KDtree\n"
                               "2. Random Forest\n")

                # For KDTree
                if clsval == "1" or clsval == 1:

                    # To include junk or to not include junk
                    junkbool = 0
                    while junkbool == 0:

                        # Select from whether to add junk or to not add junk
                        junkbool = input("Would you like to include junk as well?. Select and press enter.\n"
                                           "y/Y. Add Junk\n"
                                           "n/N. Don't add Junk\n")

                        # If we want to add Junk
                        if junkbool == "y" or junkbool == "Y":

                            # If a pickled model is found
                            if os.path.isfile("kdtree_junk_nonbonus.sav"):
                                ynbool = 0
                                while ynbool == 0:

                                    # Use this pickle or retrain and get a new pickle to use
                                    ynbool = input("A pickled model already exist for KDTree + Junk + Non-Bonus.\n"
                                                   "Would you like to use that or get a new trained model? Select and press enter\n"
                                                   "y/Y. Use the previously pickled model\n"
                                                   "n/N. Retrain the model, get a new pickle and use that\n")

                                    # If you want to use previously pickled version
                                    if ynbool == "y" or ynbool == "Y":
                                        classifyusingKDtree("nonbonus", "junk", "kdtree_junk_nonbonus.sav")

                                    # If you want to retrain a new model and pickle and use it
                                    elif ynbool == "n" or ynbool == "N":
                                        classifyusingKDtree("nonbonus", "junk", "retrain")

                                    # Invalid selection
                                    else:
                                        print("Invalid selection. Retry!")
                                        ynbool = 0

                            # If a pickled model is not found then train the model and create a pickle
                            else:
                                classifyusingKDtree("nonbonus", "junk", "retrain")

                        # If we choose to not add Junk
                        elif junkbool == "n" or junkbool == "N":

                            # If a pickled model is found
                            if os.path.isfile("kdtree_nojunk_nonbonus.sav"):
                                ynbool = 0
                                while ynbool == 0:

                                    # Use this pickle or retrain and get a new pickle to use
                                    ynbool = input("A pickled model already exist for KDTree + No Junk + Non-Bonus.\n"
                                                   "Would you like to use that or get a new trained model? Select and press enter\n"
                                                   "y/Y. Use the previously pickled model\n"
                                                   "n/N. Retrain the model, get a new pickle and use that\n")

                                    # If you want to use previously pickled version
                                    if ynbool == "y" or ynbool == "Y":
                                        classifyusingKDtree("nonbonus", "nojunk", "kdtree_nojunk_nonbonus.sav")

                                    # If you want to retrain a new model and pickle and use it
                                    elif ynbool == "n" or ynbool == "N":
                                        classifyusingKDtree("nonbonus", "nojunk", "retrain")

                                    # Invalid selection
                                    else:
                                        print("Invalid selection. Retry!")
                                        ynbool = 0

                            # If a pickled model is not found then train the model and create a pickle
                            else:
                                classifyusingKDtree("nonbonus", "nojunk", "retrain")

                        # Invalid selection
                        else:
                            print("Invalid selection. Retry!")
                            junkbool = 0


                # For Random Forest
                elif clsval == "2" or clsval == 2:

                    # To include junk or to not include junk
                    junkbool = 0
                    while junkbool == 0:

                        # Select from whether to add junk or to not add junk
                        junkbool = input("Would you like to include junk as well?. Select and press enter.\n"
                                         "y/Y. Add Junk\n"
                                         "n/N. Don't add Junk\n")

                        # If we want to add Junk
                        if junkbool == "y" or junkbool == "Y":

                            # If a pickled model is found
                            if os.path.isfile("rforest_junk_nonbonus.sav"):
                                ynbool = 0
                                while ynbool == 0:

                                    # Use this pickle or retrain and get a new pickle to use
                                    ynbool = input("A pickled model already exist for RForest + Junk + Non-Bonus.\n"
                                                   "Would you like to use that or get a new trained model? Select and press enter\n"
                                                   "y/Y. Use the previously pickled model\n"
                                                   "n/N. Retrain the model, get a new pickle and use that\n")

                                    # If you want to use previously pickled version
                                    if ynbool == "y" or ynbool == "Y":
                                        classifyusingrandomforest("nonbonus", "junk",
                                                                  "rforest_junk_nonbonus.sav")

                                    # If you want to retrain a new model and pickle and use it
                                    elif ynbool == "n" or ynbool == "N":
                                        classifyusingrandomforest("nonbonus", "junk", "retrain")

                                    # Invalid selection
                                    else:
                                        print("Invalid selection. Retry!")
                                        ynbool = 0

                            # If a pickled model is not found then train the model and create a pickle
                            else:
                                classifyusingrandomforest("nonbonus", "junk", "retrain")

                        # If we choose to not add Junk
                        elif junkbool == "n" or junkbool == "N":

                            # If a pickled model is found
                            if os.path.isfile("rforest_nojunk_nonbonus.sav"):
                                ynbool = 0
                                while ynbool == 0:

                                    # Use this pickle or retrain and get a new pickle to use
                                    ynbool = input("A pickled model already exist for RForest + No Junk + Non-Bonus.\n"
                                                   "Would you like to use that or get a new trained model? Select and press enter\n"
                                                   "y/Y. Use the previously pickled model\n"
                                                   "n/N. Retrain the model, get a new pickle and use that\n")

                                    # If you want to use previously pickled version
                                    if ynbool == "y" or ynbool == "Y":
                                        classifyusingrandomforest("nonbonus", "nojunk",
                                                                  "rforest_nojunk_nonbonus.sav")

                                    # If you want to retrain a new model and pickle and use it
                                    elif ynbool == "n" or ynbool == "N":
                                        classifyusingrandomforest("nonbonus", "nojunk", "retrain")

                                    # Invalid selection
                                    else:
                                        print("Invalid selection. Retry!")
                                        ynbool = 0

                            # If a pickled model is not found then train the model and create a pickle
                            else:
                                classifyusingrandomforest("nonbonus", "nojunk", "retrain")

                        # Invalid selection
                        else:
                            print("Invalid selection. Retry!")
                            junkbool = 0


                # Invalid selection
                else:
                    print("Invalid selection. Retry!")
                    clsval = 0


        # Invalid selection
        else:
            print("Invalid selection. Retry!")
            selval = 0


if __name__ == "__main__":
    main()