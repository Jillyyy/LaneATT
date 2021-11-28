import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('video.avi', fourcc, float(5), (360, 640))
    