#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:16:46 2019

@author: cameron
"""

from sklearn.metrics import classification_report
import pandas as pd
import math

#function for calculating facial point distances
def distance(x1, x2, y1, y2):
    xDist = x2-x1
    yDist = y2-y1
    return math.sqrt(math.pow(xDist, 2)+math.pow(yDist, 2))

class Person:
    
    #uses filename as an init arg for use in data-reading method
    def __init__(self, ident):
        self.id = ident
        self.data = pd.DataFrame
        self.features = {}
        #calls getFeatures to initialize member variables
        self.getFeatures()
    
    def getFeatures(self):
        self.getData()
        self.features['eyeLength'] = self.eyeLength()
        self.features['eyeDistance'] = self.eyeDist()
        self.features['nose'] = self.nose()
        self.features['lipSize'] = self.lipSize()
        self.features['lipLength'] = self.lipLength()
        self.features['eyebrowLength'] = self.browLength()
        self.features['aggressive'] = self.aggressive()
        
    def getData(self):
        #opens file using ident and reads in the data, starting at line 3 with columns x and y
        self.data = pd.read_csv("Face Database/"+self.id[:-3]+"/"+self.id+
                                ".csv", header=2, names=['x','y'], sep=' ')
        #drops the last row as it is not part of the data
        self.data.drop(self.data.tail(1).index,inplace=True)
        #converts the columnal data from string to floats
        self.data['x']=self.data.x.astype(float)
        self.data['y']=self.data.y.astype(float)
        
    def compare(self,other):
        sum = 0
        for key in self.features:
            #adds the square of the difference of each feature to the total sum
            sum += math.pow((self.features[key]-other.features[key]), 2)
        #returns the sqrt of the sum for euclidean distance
        return math.sqrt(sum)
        
    def eyeLength(self):
        #calculates distances for both eyes
        right = distance(self.data.loc[9,'x'],self.data.loc[10,'x'],
                         self.data.loc[9,'y'],self.data.loc[10,'y'])
        left = distance(self.data.loc[11,'x'],self.data.loc[12,'x'],
                        self.data.loc[11,'y'],self.data.loc[12,'y'])
        #uses the larger value in the ratio calculation
        if right > left:
            return right/distance(self.data.loc[8,'x'],self.data.loc[13,'x'],
                                  self.data.loc[8,'y'],self.data.loc[13,'y'])
        else:
            return left/distance(self.data.loc[8,'x'],self.data.loc[13,'x'],
                                 self.data.loc[8,'y'],self.data.loc[13,'y'])
        
    def eyeDist(self):
        return distance(self.data.loc[10,'x'],self.data.loc[11,'x'],
                        self.data.loc[10,'y'],self.data.loc[11,'y']) / distance(
                                self.data.loc[8,'x'],self.data.loc[13,'x'],
                                self.data.loc[8,'y'],self.data.loc[13,'y'])
        
    def nose(self):
        return distance(self.data.loc[15,'x'],self.data.loc[16,'x'],
                        self.data.loc[15,'y'],self.data.loc[16,'y']) / distance(
                                self.data.loc[20,'x'],self.data.loc[21,'x'],
                                self.data.loc[20,'y'],self.data.loc[21,'y'])
        
    def lipSize(self):
        return distance(self.data.loc[2,'x'],self.data.loc[3,'x'],
                        self.data.loc[2,'y'],self.data.loc[3,'y']) / distance(
                                self.data.loc[17,'x'],self.data.loc[18,'x'],
                                self.data.loc[17,'y'],self.data.loc[18,'y'])
        
    def lipLength(self):
        return distance(self.data.loc[2,'x'],self.data.loc[3,'x'],
                        self.data.loc[2,'y'],self.data.loc[3,'y']) / distance(
                                self.data.loc[20,'x'],self.data.loc[21,'x'],
                                self.data.loc[20,'y'],self.data.loc[21,'y'])
        
    def browLength(self):
        #see eyeLength
        right = distance(self.data.loc[4,'x'],self.data.loc[5,'x'],
                         self.data.loc[4,'y'],self.data.loc[5,'y'])
        left = distance(self.data.loc[6,'x'],self.data.loc[7,'x'],
                        self.data.loc[6,'y'],self.data.loc[7,'y'])
        if right > left:
            return right/distance(self.data.loc[8,'x'],self.data.loc[13,'x'],
                                  self.data.loc[8,'y'],self.data.loc[13,'y'])
        else:
            return left/distance(self.data.loc[8,'x'],self.data.loc[13,'x'],
                                 self.data.loc[8,'y'],self.data.loc[13,'y'])
            
    def aggressive(self):
        return distance(self.data.loc[10,'x'],self.data.loc[19,'x'],
                        self.data.loc[10,'y'],self.data.loc[19,'y']) / distance(
                                self.data.loc[20,'x'],self.data.loc[21,'x'],
                                self.data.loc[20,'y'],self.data.loc[21,'y'])

#First we create sample objects for the 10 people
samples = {'m1' : Person("m-001-01"), 'm2' : Person("m-002-01"), 
           'm3' : Person("m-003-01"), 'm4' : Person("m-004-01"),
           'm5' : Person("m-005-01"), 'w1' : Person("w-001-01"),
           'w2' : Person("w-002-01"), 'w3' : Person("w-003-01"),
           'w4' : Person("w-004-01"), 'w5' : Person("w-005-01")}

#Next we run a test for each test person to see which sample is the closest neighbor
tests = {'t1' : Person("m-001-05"), 't2' : Person("m-002-05"), 
         't3' : Person("m-003-05"), 't4' : Person("m-004-05"),
         't5' : Person("m-005-05"), 't6' : Person("w-001-05"),
         't7' : Person("w-002-05"), 't8' : Person("w-003-05"),
         't9' : Person("w-004-05"), 't10' : Person("w-005-05")}
y_pred = []
for k1 in tests:
    nearestNeighbor='m1'
    for k2 in samples:
        if samples[k2].compare(tests[k1]) < samples[nearestNeighbor].compare(tests[k1]):
            nearestNeighbor = k2
    y_pred.append(nearestNeighbor)
    print(k1 + "'s nearest neighbor is: " + nearestNeighbor)

y_true = []
for key in samples:
    y_true.append(key)

#finally we run a classification report
print(classification_report(y_true, y_pred))

