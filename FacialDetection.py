import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

def get_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(face) == 0:
        return img, []

    for (x, y, width, height) in face:
        cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
        roi = img[y: y+height, x: x+width]
        roi = cv2.resize(roi, (200, 200))
    return img, roi
    
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = cv2.face.LBPHFaceRecognizer_create()
model.read("finalized_model.sav")

while True:
    ret, frame = camera.read()
    img, face = get_face(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        
        if results[1] < 500:
            confidence = int(100 * (1 - (results[1] / 400)))
            disp_conf = 'Confidence: ' + str(confidence) + "%"
        cv2.putText(img, disp_conf, (150, 150), font, fontScale, color, thickness, cv2.LINE_AA)
        
        if confidence > 70:
            cv2.putText(img, "Face recognized", (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('App', img)
        else:
            cv2.putText(img, "Face not recognized", (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('App', img)
    except:
        cv2.putText(img, "No face found", (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('App', img)
    
    if cv2.waitKey(1) == 13:
        break

camera.release()
cv2.destroyAllWindows()
