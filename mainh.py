import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*
model=YOLO('yolov8l.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('video.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
cy1=280
cy2=300
offset=5
upcar={}
downcar={}
countercarup=[]
countercardown=[]
downbus={}
counterbusdown=[]
upbus={}
counterbusup=[]
uptruck={}
downtruck={}
countertruckup=[]
countertruckdown=[]

# Variable pour contrôler le saut des frames
frame_skip = 4  # Sauter une frame sur deux


while True:    
    ret,frame = cap.read()
    if not ret:
        cv2.putText(frame, "Fin de la vidéo. Appuyez sur une touche pour quitter.", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("RGB", frame)
        cv2.waitKey(0)  # Garde l'image affichée jusqu'à une touch
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]
    list1=[]
    list2=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
           list.append([x1,y1,x2,y2])
          
        elif'bus' in c:
            list1.append([x1,y1,x2,y2])
          
        elif 'truck' in c:
             list2.append([x1,y1,x2,y2])
            

    bbox_idx=tracker.update(list)
    bbox2_idx=tracker2.update(list2)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        cx3=int(x3+x4)//2
        cy3=int(y3+y4)//2
        if cy1<(cy3+offset) and cy1>(cy3-offset):
          upcar[id1]=(cx3,cy3)
        if id1 in upcar:
           if cy2<(cy3+offset) and cy2>(cy3-offset):
               
              cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
              cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
              cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
              if countercarup.count(id1)==0:
                 countercarup.append(id1)
#############################cardown##################################       
        if cy2<(cy3+offset) and cy2>(cy3-offset):
           downcar[id1]=(cx3,cy3)
        if id1 in downcar:
           if cy1<(cy3+offset) and cy1>(cy3-offset):
               
              cv2.circle(frame,(cx3,cy3),4,(255,0,255),-1)
              cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
              cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
              if countercardown.count(id1)==0:
                 countercardown.append(id1)          
##################################upbus#################################                    
    for bbox2 in bbox2_idx:
        x7,y7,x8,y8,id2=bbox2
        cx5=int(x7+x8)//2
        cy5=int(y7+y8)//2
        if cy1<(cy5+offset) and cy1>(cy5-offset):
            print(f"Camion détecté montant, ID : {id2}, centre : ({cx5}, {cy5})")
            uptruck[id2]=(cx5,cy5)
        if id2 in uptruck:
           if cy2<(cy5+offset) and cy2>(cy5-offset):
            
              cv2.circle(frame,(cx5,cy5),4,(0,255,0),-1)
              cv2.rectangle(frame,(x7,y7),(x8,y8),(255,0,255),2)
              cvzone.putTextRect(frame,f'{id2}',(x7,y7),1,1)
              if countertruckup.count(id2)==0:
                 countertruckup.append(id2)
                 print(f"Camion {id2} détecté montant, ajouté au compteur.")

#############################downbus###################################
        if cy2<(cy5+offset) and cy2>(cy5-offset):
           print(f"Camion détecté descendant, ID : {id2}, centre : ({cx5}, {cy5})")
           downtruck[id2]=(cx5,cy5)
        if id2 in downtruck:
           if cy1<(cy5+offset) and cy1>(cy5-offset):
            
              cv2.circle(frame,(cx5,cy5),4,(255,0,255),-1)
              cv2.rectangle(frame,(x7,y7),(x8,y8),(255,0,0),2)
              cvzone.putTextRect(frame,f'{id2}',(x7,y7),1,1)
              if countertruckdown.count(id2)==0:
                 countertruckdown.append(id2)
                 print(f"Camion {id2} détecté descendant, ajouté au compteur.")
                  
    cv2.line(frame,(1,cy1),(1018,cy1),(0,255,0),2)
    cv2.line(frame,(3,cy2),(1016,cy2),(0,0,255),2)
    cup=len(countercarup)
    cdown=len(countercardown)
    ctruckup=len(countertruckup)
    ctruckdown=len(countertruckdown)
    cvzone.putTextRect(frame,f'upcar:-{cup}',(50,60),2,2)
    cvzone.putTextRect(frame,f'downcar:-{cdown}',(50,160),2,2)
    cvzone.putTextRect(frame,f'truckup:-{ctruckup}',(792,43),2,2)
    cvzone.putTextRect(frame,f'truckdown:-{ctruckdown}',(792,100),2,2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows() 