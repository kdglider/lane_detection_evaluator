'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Sarjana Oradiambalam Sachidanandam
@file       AlgorithmEvaluator
@date       2020/04/28
@brief      TBD
@license    This project is released under the BSD-3-Clause license.
'''

import time
import os
import xlsxwriter

import numpy as np
import cv2

from Algorithm1 import Algorithm1

'''
@brief      
'''
class AlgorithmEvaluator:
    imageFolder = None      # Path to folder with all KITTI images
    gtFolder = None         # Path to folder with all KITTI ground truth images
    numImages = 0           # Number of images in the dataset

    algorithm = None        # Imported lane detection algorithm class object

    runTimes = None         # Array of runtimes for all test images

    passFailArray = None    # Array of pass/fail indicators for all images

    precisionArray = None   # Array of precission scores
    recallArray = None      # Array of recall scores
    fmeasureArray = None    # Array of F-measure scores
    accuracyArray = None    # Array of accuracy scores
    totalScores = None      # Array of total accuracy scores

    

    def __init__(self, Algorithm, imageFolder, gtFolder):
        self.algorithm = Algorithm(0.5)

        self.imageFolder = imageFolder
        self.gtFolder = gtFolder
        self.numImages = len(os.listdir(imageFolder))

        self.runTimes = np.zeros(self.numImages)
        self.passFailArray = np.zeros(self.numImages)

        self.precisionArray = np.zeros(self.numImages)
        self.recallArray = np.zeros(self.numImages)
        self.fmeasureArray = np.zeros(self.numImages)
        self.accuracyArray = np.zeros(self.numImages)
        self.totalScores = np.zeros(self.numImages)
    

    def evaluate(self, frame, gt, frameNumber):
        # Start timer
        start = time.perf_counter()

        # Run algorithm on frame
        output, binaryOutput = self.algorithm.detectLane(frame)

        # Stop timer
        stop = time.perf_counter()

        # Record runtime
        runTime = round(stop - start, 3)
        self.runTimes[frameNumber] = runTime
        #print(runTime)
        
        '''
        print(binaryOutput.shape)
        print(binaryOutput.dtype)

        print(gt.shape)
        print(gt.dtype)
        '''

        # Calculate true/false positives and true/false negatives
        TP = cv2.countNonZero(cv2.bitwise_and(binaryOutput, gt))
        TN = cv2.countNonZero(cv2.bitwise_not(cv2.bitwise_or(binaryOutput, gt)))
        FP = cv2.countNonZero(binaryOutput - gt)
        FN = cv2.countNonZero(gt - binaryOutput)

        '''
        print(TP)
        print(TN)
        print(FP)
        print(FN)

        cv2.imshow('output', binaryOutput)
        cv2.imshow('gt', gt)
        cv2.imshow('TP', cv2.bitwise_and(binaryOutput, gt))
        cv2.imshow('TN', cv2.bitwise_not(cv2.bitwise_or(binaryOutput, gt)))
        cv2.imshow('FP', binaryOutput - gt)
        cv2.imshow('FN', gt - binaryOutput)
        cv2.waitKey(0)
        '''


        # Calculate accuracy scores
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if (precision + recall == 0):
            fmeasure = 0
        else:
            fmeasure = 2*precision*recall / (precision + recall)
        accuracy = (TP + TN) / (TP + FP + TN + FN)

        totalScore = (precision + recall + fmeasure + accuracy) / 4
        
        '''
        print(precision)
        print(recall)
        print(fmeasure)
        print(accuracy)
        print(totalScore)
        '''

        self.precisionArray[frameNumber] = precision
        self.recallArray[frameNumber] = recall
        self.fmeasureArray[frameNumber] = fmeasure
        self.accuracyArray[frameNumber] = accuracy
        self.totalScores[frameNumber] = totalScore

        # Mark frame as "pass" if the total score is above a threshold
        self.passFailArray[frameNumber] = int(totalScore > 0.7)


    def runApplication(self, saveData=False):
        frameNumber = 0

        for imageName, gtName in zip(os.listdir(self.imageFolder), os.listdir(self.gtFolder)):
            image = cv2.imread(self.imageFolder + imageName)
            gt = cv2.imread(self.gtFolder + gtName, 0)

            # Ensure the image and ground truth are the same size
            gt = cv2.resize(gt, (image.shape[1], image.shape[0]))

            #print(frameNumber)

            self.evaluate(image, gt, frameNumber)

            frameNumber += 1
        
        if (saveData == True):
            workbook = xlsxwriter.Workbook('data.xlsx')
            worksheet = workbook.add_worksheet()
            row = 0
            col = 0

            worksheet.write(row, col,    'Run Time (s)')
            worksheet.write(row, col+1,  'Precision')
            worksheet.write(row, col+2,  'Recall')
            worksheet.write(row, col+3,  'F-Measure')
            worksheet.write(row, col+4,  'Accuracy')
            worksheet.write(row, col+5,  'Total Accuracy Score')

            row += 1

            for runTime, precision, recall, fmeasure, accuracy, totalScore \
                in zip(self.runTimes, 
                       self.precisionArray,
                       self.recallArray,
                       self.fmeasureArray,
                       self.accuracyArray,
                       self.totalScores):
                
                worksheet.write(row, col,    runTime)
                worksheet.write(row, col+1,  precision)
                worksheet.write(row, col+2,  recall)
                worksheet.write(row, col+3,  fmeasure)
                worksheet.write(row, col+4,  accuracy)
                worksheet.write(row, col+5,  totalScore)

                row += 1
            
            workbook.close()


if __name__ == '__main__':
    # Path to KITTI image folder
    imageFolder = 'dataset/images/'      

    # Path to KITTI ground truth folder
    gtFolder = 'dataset/ground_truth/'

    # Flag to save data
    saveData = True

    frame = cv2.imread(imageFolder + 'um_000084.png')
    gt = cv2.imread(gtFolder + 'um_lane_000084.png', 0)

    # Run application
    evaluator = AlgorithmEvaluator(Algorithm1, imageFolder, gtFolder)
    evaluator.runApplication(saveData)

    #evaluator.evaluate(frame, gt, 0)



    
