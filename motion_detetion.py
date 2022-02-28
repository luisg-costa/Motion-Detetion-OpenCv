import cv2, datetime, pandas

video = cv2.VideoCapture(0)

first_frame = None

status_list = [0]

times = []

df = pandas.DataFrame(columns=["Start","End"])

while True:

    status = 0
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    if first_frame is None:
        first_frame = gray_frame_blur
        continue
    
    delta_frame = cv2.absdiff(first_frame, gray_frame_blur)
    thresh_delta = cv2.threshold(delta_frame, 20, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)
    
    (cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    
    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.datetime.now())
    elif status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.datetime.now())
        
    cv2.imshow("Motion Detection", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.datetime.now())
        break

for i in range(0,len(times),2):
    df = df.append({"Start":times[i],"End":times[i+1]}, ignore_index=True)    

df.to_csv("Times.csv")

video.release()

cv2.destroyAllWindows()