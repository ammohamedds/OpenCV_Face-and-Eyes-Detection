import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

videcap =cv2.VideoCapture(0)

while True:
    ret, img = videcap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, 1.1, 5)
    print(faces)
    for (x, y, w, h) in faces:  # in case of many faces and eyes
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # strating point(x,y) and ending points (x + w, y + h)
        eye_gray = imgGray[y:y + h, x:x + w]  # we will put limit for eyes inside face to avoid detecting any fake or eyesball
        eye_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(eye_gray)  # we can as 1.1, 5
        print(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

    if ret == True:     # We can remove it 
        cv2.imshow("Output Video",img)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break 