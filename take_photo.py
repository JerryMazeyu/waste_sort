def take_photo(save_path='./cache/input/'):
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        sucess, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", gray)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord("s"):
            cv2.imwrite(save_path + "input.jpg", img)
            cv2.destroyAllWindows()
            break
    cap.release()
