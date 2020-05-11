'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Sarjana Oradiambalam Sachidanandam
@file       DataSorter.py
@date       2020/05/06
@brief      Extracts marked lane images and their ground truths from the KITTI dataset
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np 
import cv2
import os

'''
@brief      Extract urban-marked and urban-marked-multilane images from collective dataset
@param      sourceFolder        Path to folder with all KITTI images
@param      destinationFolder   Path of destination folder to put marked images in
@param      imageListFile       Text file with all marked image file names
@return     None
'''
def getMarkedImages(sourceFolder, destinationFolder, imageListFile):
    '''
    Create Python list of image names from imageListFile
    Loop through all files (use os.listdir) in sourceFolder and save them in destinationFolder
    if the name appears in the list
    '''
    pass


'''
@brief      Extract marked image ground truths from collective dataset and convert them to binary
@param      sourceFolder        Path to folder with all KITTI ground truth images
@param      destinationFolder   Path of destination folder to put ground truth images in
@param      imageListFile       Text file with all marked image file names
@return     None
'''
def getMarkedGTImages(sourceFolder, destinationFolder, imageListFile):
    '''
    Create Python list of image names from imageListFile
    Loop through all files (use os.listdir) in sourceFolder 
        if the name appears in the list
            Find magenta lane area and paint it white; paint the remainder black
            Save new binary image in destinationFolder
    '''
    pass
        
    

if __name__ == '__main__':
    # Path to KITTI image folder
    imageSourceFolder = ''      

    # Path to KITTI ground truth folder
    gtSourceFolder = ''      

    # Text file with all marked image file names
    imageListFile = '.txt'

    # Destination folder for marked images
    imageDestFolder = 'dataset/images/'

    # Destination folder for marked image ground truths
    gtDestFolder = 'dataset/imageGTs/'

    getMarkedImages(imageSourceFolder, imageDestFolder, imageListFile)
    getMarkedGTImages(gtSourceFolder, gtDestFolder, imageListFile)




    
