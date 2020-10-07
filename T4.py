import numpy as np
import cv2
import matplotlib.pyplot as plt

#to use the computer camera
cap = cv2.VideoCapture(0)
#somethings to do to the video image
#width
cap.set(3,640)
#height
cap.set(4,480)
#brightness
cap.set(10,150)


def find_histogram(labels):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    #len(np.unique(labels))+1
    #nlabels = np.arange(0,len(np.unique(labels))+1)
    nlabels = 5
    (hist, _) = np.histogram(labels, bins=nlabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50,300,3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        #plot the relative percentage of each cluster
        endX = startX + (percent*300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX),50), color.astype("uint8").tolist(),-1)
        startX = endX
    #return the bar chart
    return bar


while(True):

        #capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (155,115),(485,365),(0,255,0),2)

    rec = frame[120:360,160:480] #height first! then the width
    colRec = rec.reshape((-1,3))

    #convert to np.float32
    colRec = np.float32(colRec)

    #define criteria, number of clusters (K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K=8
    ret2, label, center = cv2.kmeans(colRec,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


    hist = find_histogram(label)
    bar = plot_colors2(hist, center)

    cv2.imshow('video', frame)
    cv2.imshow('bar', bar)




    #Now convert back into uint8, and make original image
    #center = np.uint8(center)
    #res = center[label.flatten()]
    #res2 = res.reshape((rec.shape))



    #cv2.imshow('res2', res2)
    #cv2.imshow('centers', center)


    #cv2.imshow('video', frame)
    #cv2.imshow('Cropped Video', rec)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #when everything done, release the capture 
cap.release()
cv2.destroyAllWindows()