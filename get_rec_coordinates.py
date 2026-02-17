import cv2

def get_rectangles_xy(image):
    def draw_rectangle(event, x, y, flags, param):
        nonlocal rectangles, image, drawing, start_x, start_y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rectangles.append(((start_x, start_y), (x, y)))
            cv2.rectangle(image, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", image)

    # Create a window and set the callback function
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", draw_rectangle)

    rectangles = []
    drawing = False
    start_x, start_y = 0, 0

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            rectangles = []

        elif key == ord("s"):
            # Save the coordinates
            cv2.destroyAllWindows()
            return rectangles

        elif key == ord("q"):
            cv2.destroyAllWindows()
            break

    return rectangles







