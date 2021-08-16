import cv2,pandas
from datetime import datetime

first_frame=None
frame_list=[None,None]
d=[]
#that's the reference frame

video=cv2.VideoCapture(0)#turns the camera on in color mode

df=pandas.DataFrame(columns=['start','end'])
while True:
    status=0
    check,frame=video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0) 
    #increases accuracy of img. dont know how

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    #compares current frame with the first frame

    td=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    td=cv2.dilate(td,None,iterations=2)#smoothens the white area

    (cnts,_)=cv2.findContours(td.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            status=1
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    frame_list.append(status)
    if frame_list[-2]==0 and frame_list[-1]==1:
        d.append(datetime.now())
    #time at which object enters the frame

    if frame_list[-2]==1 and frame_list[-1]==0:
        d.append(datetime.now())
    #time at which object exits the frame


    cv2.imshow("MJ",gray)
    cv2.imshow("difference",delta_frame)
    cv2.imshow('threshold frame',td)
    cv2.imshow('contour frame',frame)
    

    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            d.append(datetime.now())
        break

print(frame_list)
print(d)
print(len(d))
video.release()
cv2.destroyAllWindows


for i in range(0,len(d),2):
    df=df.append({'start':d[i],'end':d[i+1]},ignore_index=True)

df.to_csv('times.csv')
