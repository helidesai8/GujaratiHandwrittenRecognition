import cv2
import numpy as np


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


# Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    print("Reading image..")
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255 - img_bin

    print("Storing binary image to Images/Image_bin.jpg..")
    cv2.imwrite("Images/Image_bin.jpg", img_bin)

    print("Applying Morphological Operations..")
    kernel_length = np.array(img).shape[1] // 40

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    print("Binary image which only contains boxes: Images/img_final_bin.jpg")
    cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    print("Output stored in Output directiory!")

    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 22 and h > 22) and w >= h:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cv2.imwrite("./Output/" + str(idx) + '.png', new_img)

    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("./Temp/img_contour.jpg", img)


box_extraction("sample.jpg", "./Output/")