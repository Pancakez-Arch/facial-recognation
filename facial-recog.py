import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


cap = cv2.VideoCapture(0)

while True:
 
    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
       
        roi_gray = gray[y:y+h, x:x+w]
        
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        for (sx, sy, sw, sh) in smiles:
            
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
       
            print("Smile detected! Sending command to swarm...")



    cv2.imshow('Smile Detection', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

