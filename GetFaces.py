import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)

# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

def get_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(face) == 0:
        return None

    for (x, y, width, height) in face:
        cropped_face = img[y: y+height, x: x+width]
    
    return cropped_face

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

while True:
    ret, frame = camera.read()
    if get_face(frame) is not None:
        count += 1
        face = cv2.resize(get_face(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Save images
        file_path = './faces/' + str(count) + '.jpg'
        cv2.imwrite(file_path, face)
        
        cv2.putText(face, str(count), org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Face', face)
    else:
        print('face not found')
    
    if cv2.waitKey(1) == 13 or count == 200:
        break

camera.release()
cv2.destroyAllWindows()
print('Complete')
