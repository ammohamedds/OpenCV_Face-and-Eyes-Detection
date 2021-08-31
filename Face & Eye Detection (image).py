import cv2
# It is sensitive to noise, so it is accurate with complex images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread("Images/people.jpg")
# img = cv2.imread("Images/5.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)
print(f'Number of faces found = {len(faces)}')
print(faces)
# When no faces detected, face_classifier returns and empty tuple
# if faces is ():
#     print("No faces found")

for (x, y, w, h) in faces:  # in case of many faces and eyes
    cv2.rectangle( img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # strating point(x,y) and ending points (x + w, y + h)
    eye_gray = imgGray[y:y+h, x:x+w]  # we will put limit for eyes inside face to avoid detecting any fake or eyesball
    eye_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(eye_gray)  # we can as 1.1, 5
    print(eyes)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)


    # cv2.imwrite("Face", img)
cv2.imshow("Face2", img)

cv2.waitKey(0)
cv2.imwrite("Images/Faces Detection.png", img)
