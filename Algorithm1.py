'''
Copyright (c) 2020 Hao Da (Kevin) Dong
@file       Algorithm1.py
@date       2020/03/08
@brief      Lane detection application
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2

'''
@brief      Application which detects and overlays road lanes in a video stream
'''
class Algorithm1:
    K = np.array([])                    # Camera intrinsic matrix
    distortionCoeffs = np.array([])     # Camera distortion coefficients

    cropFactor = 0      # 0-1 value that represents percentage to crop from the top of a frame to isolate lanes
    cropLineY = 0       # cropFactor value in terms of pixels from the top of the frame

    frame = np.array([])        # Current
    laneMask = np.array([])

    yellowHSVLowBound = np.array([10, 70, 200])
    yellowHSVUpperBound = np.array([65, 180, 255])

    whiteHSVLowBound = np.array([0, 0, 190])
    whiteHSVUpperBound = np.array([170, 25, 255])


    def __init__(self, K, distortionCoeffs, cropFactor):
        self.K = K
        self.distortionCoeffs = distortionCoeffs
        self.cropFactor = cropFactor


    '''
    @brief      Undistort an image, remove noise and crop to ROI
    @param      
    @return     
    '''
    def prepareImage(self, frame):
        h = frame.shape[0]
        w = frame.shape[1]

        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.distortionCoeffs, \
                                                 (w, h), 0, (w, h))
        undistortedFrame = cv2.undistort(frame, self.K, self.distortionCoeffs, None, newK)
        
        self.frame = undistortedFrame

        blurredFrame = cv2.GaussianBlur(undistortedFrame, (5,5), 0)

        self.cropLineY = int(self.cropFactor*h)
        croppedFrame = blurredFrame[self.cropLineY:h, :, :]
        
        return croppedFrame


    '''
    @brief      Determine the lane line equations from a frame
    @param      
    @return     
    '''
    def getLaneLines(self, frame):
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        yellowMask = cv2.inRange(hsvFrame, self.yellowHSVLowBound, self.yellowHSVUpperBound) 
        whiteMask = cv2.inRange(hsvFrame, self.whiteHSVLowBound, self.whiteHSVUpperBound)

        self.laneMask = yellowMask | whiteMask

        lines = cv2.HoughLinesP(self.laneMask, 1, np.pi / 180, 50)

        lineParams = []
        if (lines is not None and len(lines) > 1):
            for i in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[i]:

                    #cv2.line(undistortedFrame,(x1,y1+cropLineY),(x2,y2+cropLineY),(0,0,255),2)
                    
                    if (y2 != y1 and x2 != x1):
                        m = (y2-y1) / (x2-x1)
                        b = y1 - m*x1
                        if (np.isnan(m) == False and np.isnan(b) == False):
                            lineParams.append([m,b])
                    else:
                        continue              

        lineParams = np.array(lineParams)    

        if (len(lineParams) <= 1):
            return lineParams

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Apply KMeans
        lineParams = np.float32(lineParams)
        _, _, centers = cv2.kmeans(lineParams, 2, None, criteria, 10, flags)

        m1 = centers[0,0]
        m2 = centers[1,0]
        b1 = centers[0,1] + self.cropLineY
        b2 = centers[1,1] + self.cropLineY

        laneParams = np.array([[m1, b1],
                               [m2, b2]])
        
        return laneParams


    '''
    @brief      Display the lane lines, lane mesh and drivetrajectory angle on the frame
    @param      
    @return     
    '''
    def visualization(self, laneParams):
        if (len(laneParams) <= 1):
            return self.frame

        h = self.frame.shape[0]
        w = self.frame.shape[1]

        m1 = laneParams[0,0]
        m2 = laneParams[1,0]
        b1 = laneParams[0,1]
        b2 = laneParams[1,1]

        y1 = h
        y2 = self.cropLineY
        y3 = h
        y4 = self.cropLineY

        x1 = int((y1 - b1) / m1)
        x2 = int((y2 - b1) / m1)
        x3 = int((y3 - b2) / m2)
        x4 = int((y4 - b2) / m2)

        outputFrame = self.frame.copy()

        # Draw lane lines
        cv2.line(outputFrame, (x1,y1), (x2,y2), color=(0,255,0), thickness=3)
        cv2.line(outputFrame, (x3,y3), (x4,y4), color=(0,255,0), thickness=3)

        # Create a translucent mesh over the lane
        overlay = outputFrame.copy()
        laneMesh = np.array([[x1,y1], [x2,y2], [x4,y4], [x3,y3]])

        cv2.fillPoly(overlay, [laneMesh], color=(0, 0, 255))
        cv2.addWeighted(src1=overlay, alpha=0.5, \
                        src2=outputFrame, beta=0.5, \
                        dst=outputFrame, gamma=0)
        
        # Calculate the centerline angle and display the driving trajectory
        laneAngle1 = np.degrees(np.arctan(m1))
        laneAngle2 = np.degrees(np.arctan(m2))
        if (laneAngle1 < 0):
            laneAngle1 = laneAngle1 + 180
        if (laneAngle2 < 0):
            laneAngle2 = laneAngle2 + 180

        centerlineAngle = (laneAngle1 + laneAngle2) / 2
        
        textOverlay = 'Drive Trajectory: ' \
                    + str(round(centerlineAngle - 90)) \
                    + ' degrees from center'

        cv2.putText(outputFrame, textOverlay, (10,30), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, \
                    color=(0,0,255), thickness=3)

        
        '''scale = 1000 / w  # percent of original size
        dim = (int(w * scale), int(h * scale))
        dimCropped = (int(self.laneMask.shape[1] * scale), int(self.laneMask.shape[0] * scale))

        cv2.imshow("Frame", cv2.resize(outputFrame, dim))
        cv2.imshow("Mask", cv2.resize(self.laneMask, dimCropped))
        cv2.waitKey(0)'''
        

        return outputFrame


    '''
    @brief      Demonstration function to run the entire lane detection application with video
    @param      
    @return     
    '''
    def runApplication(self, videoFile, saveVideo=False):
        # Create video stream object
        videoCapture = cv2.VideoCapture(videoFile)
        # videoCapture.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        
        # Define video codec and output file if video needs to be saved
        if (saveVideo == True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 720p 30fps video
            out = cv2.VideoWriter('LaneDetectionOutput.mp4', fourcc, 30, (1280, 720))

        # Continue to process frames if the video stream object is open
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()

            # Continue processing if a valid frame is received
            if ret == True:
                # Detect and visualize lanes
                preparedFrame = laneDetector.prepareImage(frame)
                laneParams = laneDetector.getLaneLines(preparedFrame)
                outputFrame = laneDetector.visualization(laneParams)

                #cv2.imwrite('sampleLane2.png', frame)

                # Save video if desired, resizing frame to 720p
                if (saveVideo == True):
                    out.write(cv2.resize(outputFrame, (1280, 720)))
                
                # Display frame to the screen in a video preview
                cv2.imshow("Frame", cv2.resize(outputFrame, (1280, 720)))

                # Exit if the user presses 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # If the end of the video is reached, wait for final user keypress and exit
            else:
                cv2.waitKey(0)
                break
        
        # Release video and file object handles
        videoCapture.release()
        if (saveVideo == True):
            out.release()
        
        print('Video and file handles closed')



if __name__ == '__main__':
    
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                   [0.000000e+00, 9.019653e+02, 2.242509e+02],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    distortionCoeffs = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    '''
    K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                  [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    distortionCoeffs = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    '''

    # How much to crop a frame from the top to isolate the road
    cropFactor = 0.5

    # Select video file
    videoFile = 'sample_data1/video1.mp4'

    # Choose whether or not to save the output video
    saveVideo = False

    # Run application
    laneDetector = Algorithm1(K, distortionCoeffs, cropFactor)
    #laneDetector.runApplication(videoFile, saveVideo)

    
    frame = cv2.imread('sampleLane.png')
    preparedFrame = laneDetector.prepareImage(frame)
    laneParams = laneDetector.getLaneLines(preparedFrame)
    outputFrame = laneDetector.visualization(laneParams)
    
    

