import cv2
import pytesseract
import imutils
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
framewidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)

cap.set(3, framewidth)
cap.set(4, frameheight)

while True:
    success, img = cap.read( )
    
    # Greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral Filter
    blf_img = cv2.bilateralFilter(grey_img, 11, 17, 17)
    
    # Canny Edge Detection
    ced_img = cv2.Canny(blf_img, 30, 200)
    
    cnts = cv2.findContours(ced_img.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                  screenCnt = approx
                  #break
    if screenCnt is None:
               detected = 0

    else:
               detected = 1
    if detected == 1:
               cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)    
    
    mask = np.zeros(grey_img.shape,np.uint8)
    if screenCnt is not None:
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img,img,mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = grey_img[topx:bottomx+1, topy:bottomy+1]
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        text111 = text
        text111 = text111[:2] + text111[3:5] + text111[6:]
        print("Detected Number is:",text)
        cv2.imshow("Frame", img)
        cv2.imshow('Cropped',Cropped)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
    