import cv2
import mediapipe as mp
import pyautogui
import numpy as np
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

wCam, hCam = 640, 480
frameR = 200 #frame
# reduction


while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)


    if landmark_points:
        landmarks = landmark_points[0].landmark


        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

            if id == 1:
                screen_x = np.interp(x, (frameR, wCam-frameR), (0, screen_w))
                screen_y = np.interp(y, (frameR,hCam-frameR), (0, screen_h))
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        mouth = [landmarks[11], landmarks[14]]
        for landmark in mouth:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if(mouth[1].y - mouth[0].y) > 0.020:
            pyautogui.click()
            pyautogui.sleep(1)


    cv2.imshow('Eye Controller Mouse', frame)
    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()
cam.release()

