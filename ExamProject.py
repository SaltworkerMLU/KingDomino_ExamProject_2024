import cv2
import numpy as np

file = 'Examples/figures/2.jpg'
img = cv2.imread(file)
img_gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

ret, img_blue = cv2.threshold(img_hsv[:,:,0], 100, 255, cv2.THRESH_BINARY)
ret, img_yellow = cv2.threshold(img[:,:,2], 175, 255, cv2.THRESH_BINARY) # Alt: #ret, img_yellow = cv2.threshold(img[:,:,2], 175, 255, cv2.THRESH_BINARY)
ret, img_yg = cv2.threshold(img[:,:,1], 127, 255, cv2.THRESH_BINARY)
ret, img_black = cv2.threshold(img[:,:,1], 40, 255, cv2.THRESH_BINARY)
ret, img_desert = cv2.threshold(img[:,:,0], 50, 255, cv2.THRESH_BINARY)
ret, img_forest = cv2.threshold(img[:,:,2], 40, 255, cv2.THRESH_BINARY_INV)
img_edge = cv2.Canny(img_gray, 50, 200)

def mean_filter(image, size):
    kernel = np.full((size, size), 1 / size**2)

    filter_image = cv2.filter2D(image, -1, kernel)

    #filter_image = cv2.cvtColor(filter_image, cv2.IMREAD_GRAYSCALE)

    return filter_image

def thresholds(img):
    kernel = cv2.imread('Examples/figures/crown.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    base = np.zeros([len(img), len(img[0])])

    for x, row in enumerate(img): # range(len(img))
        for y, uintpixel in enumerate(row): # range(len(img[0]))
            pixel = uintpixel.astype(np.float64) # /255

            #print(pixel[1])

            """if pixel[1] < 100 and pixel[2] > 100:
                base[x, y] = 255
            else:
                base[x, y] = 0"""
            
            if pixel[2] < 100 and pixel[2] > 70:
                base[x, y] = 255
            else:
                base[x, y] = 0
            
            """if pixel[0] > 50 and pixel[2] > 100:
                base[x, y] = 255
            else:
                base[x, y] = 0"""
            
            """if pixel[1] < 40 and pixel[0] > 20:
                base[x, y] = 1
            else:
                base[x, y] = 0
    
    #temp = cv2.matchTemplate(base, kernel, cv2.TM_CCOEFF_NORMED)
    # Apply filter
    temp = cv2.filter2D(base, -1, kernel)"""

    return base # , temp

#img_edge_mean = cv2.GaussianBlur(img_edge, (51, 51), 0)
#img_edge_median = cv2.medianBlur(img_edge, 9)

#img_desert = thresholds(img)

#img_forest = thresholds(img_hsv)

"""
Detect blue square
"""

img_blue_square = cv2.medianBlur(img_blue, 25)
img_yellow_square = cv2.medianBlur(img_yellow, 25)


#cv2.imshow("RGB", img)

#cv2.imshow("Saturation", img_hsv[:,:,1])
#cv2.imshow("Base", img_base)
#cv2.imshow("Base Result", img_base_res)
#cv2.imshow("LAB", img_lab[:,:,2].astype(np.float64)/255)
#cv2.imshow("Blue", img_blue)
#cv2.imshow("Yellow", img_yellow)

img_yg = cv2.medianBlur(img_yg - img_yellow, 25)
img_desert = cv2.medianBlur(img_desert - img_blue - img_yg, 25)
img_forest = cv2.medianBlur(img_forest - img_blue, 25)
ret, img_mines = cv2.threshold(img_hsv[:,:,2], 30, 255, cv2.THRESH_BINARY_INV) # ALT: ret, img_mines = cv2.threshold(img[:,:,0] + img_hsv[:,:,1], 75, 255, cv2.THRESH_BINARY_INV)
img_mines -= img_forest + img_yg
ret, img_hue_high = cv2.threshold(img_hsv[:,:,0], 40, 255, cv2.THRESH_BINARY)
img_forest = img_hue_high - img_yg - img_blue

cv2.imshow("Blue squares", img_blue_square)
cv2.imshow("Yellow squares", img_yellow_square)
cv2.imshow("Green squares", img_yg)
#cv2.imshow("mines", img_mines)
cv2.imshow("Forest", img_forest)

ret, img_mines2 = cv2.threshold(img[:,:,1] - img[:,:,0], 10, 255, cv2.THRESH_BINARY_INV)
ret, img_desert = cv2.threshold(img[:,:,1] - img[:,:,0], 50, 255, cv2.THRESH_BINARY_INV)

#img_mines2 = cv2.medianBlur(img_mines2 - img_forest - img_yg, 25)
img_desert = cv2.medianBlur(img_desert, 25)

cv2.imshow("desert", img_desert - img_mines2 - img_forest)
cv2.imshow("mines", img_mines2)

def patternRecognizer(image, kernel):
    patternMatch = cv2.matchTemplate(image, kernel, cv2.TM_CCOEFF_NORMED)
    patternMatch = cv2.cvtColor(patternMatch, cv2.IMREAD_GRAYSCALE)

    # 8-bit provided in percentage
    ret, result = cv2.threshold(patternMatch, 0.4, 1, cv2.THRESH_BINARY)

    return result # patternMatch  #

"""img_crown = cv2.imread("Examples/figures/crown_green.png")
img_crown = cv2.rotate(img_crown, cv2.ROTATE_180)
#img_crown = cv2.cvtColor(img_crown, cv2.IMREAD_GRAYSCALE)

img_crown = patternRecognizer(img_hsv, img_crown)



cv2.imshow("V", img_hsv[:,:,1])
cv2.imshow("Crown", img_crown)"""
#cv2.imshow("Hi", img_crown)

#cv2.imshow("IMG", img_hue_high)

#cv2.imshow("Desert", img_desert)
#cv2.imshow("Edge", img_edge)
#cv2.imshow("Edge mean", img_edge_mean)


#cv2.imshow("Edge median", img_edge_median)

cv2.waitKey(0)