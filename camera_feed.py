import cv2
import time
import keyboard
from deepface import DeepFace

age_weights = "models/age_deploy.prototxt"
age_config = "models/age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights) 

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 

gender_weights = "models\gender_deploy.prototxt"
gender_config = "models/gender_net.caffemodel"
gender_Net = cv2.dnn.readNet(gender_config, gender_weights)

genderList = ["Male", "Female"]
# Hello
#emotion_Net = cv2.dnn.readNetFromONNX("models\emotion-ferplus-8.onnx")

#emotionList = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

model_mean = (78.4263377603, 87.7689143744, 114.895847746) 




face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(1)

#uzeo određeni model kojim je iz tog lica izvadil godine, spol i emociju, standard sličica. HSEmotion, Mediapipe
#https://drive.google.com/drive/folders/16qqswNHvUCGQI4iCekXdd6T_-ePKZrzz
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


while True:

    result, video_frame = video_capture.read() 
    if result is False:
        break 

    faces = detect_bounding_box(video_frame)
    
    if keyboard.is_pressed("r"):
        video_frame[:,:,1] = 0
        video_frame[:,:,0] = 0
        #for x in range(0,480):
         #   for y in range(0, 640):
          #      c = video_frame[x,y][2]
           #     video_frame[x,y] = [0, 0, c]
    if keyboard.is_pressed("g"):
        video_frame[:,:,0] = 0
        video_frame[:,:,2] = 0
        #if keyboard.is_pressed("g"):
         #   for x in range(0,480):
          #      for y in range(0, 640):
           #         c = video_frame[x,y][1]
            #        video_frame[x,y] = [0, c, 0]
    if keyboard.is_pressed("b"):
        video_frame[:,:,1] = 0
        video_frame[:,:,2] = 0
        #if keyboard.is_pressed("b"):
         #   for x in range(0,480):
          #      for y in range(0, 640):
           #         c = video_frame[x,y][0]
            #        video_frame[x,y] = [c, 0, 0]
    cv2.waitKey(10)
    if str(type(faces)) == "<class 'numpy.ndarray'>":
        box = [faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]]
        face_frame = video_frame[box[1]:box[3], box[0]:box[2]]

        blob = cv2.dnn.blobFromImage(face_frame, 1.0, (227, 227), model_mean, swapRB=False) 

        age_Net.setInput(blob) 
        age_preds = age_Net.forward()
        age = ageList[age_preds[0].argmax()] 

        cv2.putText(video_frame, f'{age}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA) 

        gender_Net.setInput(blob)
        gender_preds = gender_Net.forward()
        gender = genderList[gender_preds[0].argmax()]
        cv2.putText(video_frame, gender, ((box[0], box[3] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(video_frame, emotion, ((box[2]-100, box[3] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        

    #video_frame = video_frame[100:200, 100:200]q
    cv2.imshow("My Face Detection Project", video_frame) 
    #print(time.time())
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()