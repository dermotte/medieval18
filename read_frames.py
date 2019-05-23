"""
Simple class for reading a video from the GameStory data and searching to find a single image within it.
Used to find the logo based transitions in the stream.
by Mathias Lux
"""
import math

import cv2
import numpy as np

# video_path = "/home/mlux/"
# video = 'out.mp4'
video_path = "/media/mlux/Volume/DataSets/GameStory2018/gamestory18-data/train_set/"
# video_path = "/media/mlux/Volume/DataSets/GameStory2018/gamestory18-data/test_set/"
video = '2018-03-04_P11.mp4'
frame_number = 0
# hog = cv2.HOGDescriptor()


def create_hist(image):
    # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # hist = hog.compute(image)
    return hist

def compare_hist(h1, h2):
    d = np.linalg.norm(h1 - h2)
    # d = cv2.compareHist(h1, h2, cv2.HISTCMP_KL_DIV)
    return d

def frame2time(frame_number):
    secs = math.floor(frame_number / 59.97)
    m = math.floor(secs / 60)
    h = math.floor(m / 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m % 60, secs % (60))

if __name__ == '__main__':
    # load reference image and make histogram ...
    # frame needs to be taken from the actual video!
    image = cv2.imread('images/examples/intel_logo_cut_03.png')
    hist = create_hist(image)
    f = open(video + ".logo.l2.txt", "w")

    # cap = cv2.VideoCapture(video_path + 'out.mp4')
    cap = cv2.VideoCapture(video_path + video)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # analyze frame:
            h = create_hist(frame)
            d = compare_hist(h, hist)
            if d < 0.2:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                print("{:09d}\t{:2.3f}\t".format(frame_number, d)+frame2time(frame_number))
                f.write("{:09d}\t{:2.3f}\t{}\n".format(frame_number, d, frame2time(frame_number)))
                # Press Q on keyboard to  exit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            frame_number = frame_number + 1
            if frame_number%10000 == 0:
                print(frame2time(frame_number-1))
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    f.close()

    # Closes all the frames
    cv2.destroyAllWindows()