import numpy as np
import cv2
import glob
from scoreCalculator import *

# Define 5x5 grid colors and crowns
score_colors = [''] * 25
score_crowns = [''] * 25

# Extract files from each of the folders containing dominos, castles, and games
domino_paths = sorted(glob.glob("King Domino dataset/unique dominos/*.jpg"))
castle_paths = sorted(glob.glob("King Domino dataset/unique dominos/castles/*.jpg"))
game_paths = sorted(glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg"))

# Colors of each domino from left to right
# 'Y' = yellow; 'F' = forest; 'B' = blue; 'D' = desert; 'G' = green; 'M' = mine
domino_color_set = ['YY', 'YY', 'FF', 'FF', 'FF', 'FF',
                    'BB', 'BB', 'BB', 'GG', 'GG', 'DD', 
                    'YF', 'YB', 'YG', 'YD', 'FB', 'FG', 
                    'YF', 'YB', 'YG', 'YD', 'YM', 'FY',
                    'FY', 'FY', 'FY', 'FB', 'FG', 'BY',
                    'BY', 'BF', 'BF', 'BF', 'BF', 'YG',
                    'BG', 'YD', 'GD', 'MY', 'YG', 'BG',
                    'YD', 'GD', 'MY', 'DM', 'DM', 'YM']

# Crowns on each domino from left to right
domino_crown_set = ['00'] * 18 + ['10'] * 17 + ['01'] * 4 + ['10'] + ['02'] * 4 + ['20'] + ['02'] * 2 + ['03']

# [file, [x, y, RGB]]
domino = []
print("Loading all 48 unique dominos...")
for i in range(len(domino_paths)):
    print(i)
    temp = cv2.imread(domino_paths[i])
    domino.append( cv2.resize(temp, (200, 100)) )

# [file, [x, y, RGB]]
castle = []
print("Loading all 4 unique castles...")
for i in range(len(castle_paths)):
    print(i)
    temp = cv2.imread(castle_paths[i])
    castle.append( cv2.resize(temp, (100, 100)) )

# [file, [x, y, RGB]]
game = []
print("Loading dataset of 500x500 Kingdomino games...")
for i in range(len(game_paths)):
    print(i)
    temp = cv2.imread(game_paths[i])
    game.append( cv2.resize(temp, (500, 500)) )

kernel = domino[47] # cv2.imread("King Domino dataset/MLU database/unique dominos/36.jpg")

thresh = [0.75] * 45 + [0.45] * 2 + [0.75]

img = game[1] # cv2.imread("Examples/figures/2.jpg")

for i in range(len(domino)):
#for i in range(44, 48): # range(18, len(domino)):
    kernel = domino[i]
    for j in range(4):
        result = patternRecognizer(img, kernel, thresh[i])
        coord_match = [result.argmax(axis=0).argmax(axis=0)[0], result.argmax(axis=1).argmax(axis=0)[0]]

        # If no match was found
        if coord_match == [0, 0]:
            print("No")
        else:
            print(coord_match)

            # Start off by rounding up
            coord_match = [int( round(coord_match[0] / 50.0, 0) * 50 ), 
                            int( round(coord_match[1] / 50.0, 0) * 50 )]

            score_grid = [coord_match[0] // 100, coord_match[1] // 100]
            print(score_grid)

            # Set color and crown(s) for each matched grid
            score_colors[score_grid[0] + score_grid[1] * 5] = domino_color_set[i][0]
            score_crowns[score_grid[0] + score_grid[1] * 5] = domino_crown_set[i][0]
            if j % 2 == 0:
                score_colors[score_grid[0] + score_grid[1] * 5 + 1] = domino_color_set[i][1]
                score_crowns[score_grid[0] + score_grid[1] * 5 + 1] = domino_crown_set[i][1]
            elif j % 2 != 0:
                score_colors[score_grid[0] + score_grid[1] * 5 + 5] = domino_color_set[i][1]
                score_crowns[score_grid[0] + score_grid[1] * 5 + 5] = domino_crown_set[i][1]

            # For each pixel the domino consists of in matched image
            for height in range(100):
                for length in range(200):
                    # Set pixel in image to black
                    if j % 2 == 0:
                        img[coord_match[1] + height, coord_match[0] + length] = [0, 0, 0]
                    elif j % 2 != 0:
                        img[coord_match[1] + length, coord_match[0] + height] = [0, 0, 0]
                

            cv2.imshow("Image", img)
            cv2.imshow("Kernel", kernel)
            cv2.imshow("Pattern", result)
            cv2.waitKey(0)
            break # each unique domino can only occur once per 5x5 plate

        # Rotate 90 degrees
        kernel = cv2.rotate(kernel, cv2.ROTATE_90_COUNTERCLOCKWISE)

print(score_colors)
print(score_crowns)

score = kingdomino_scoreCalculator(score_colors, score_crowns)
print(score)

"""cv2.imshow("Image", img)
#cv2.imshow("Kernel", kernel)
#cv2.imshow("Pattern", result)
cv2.waitKey(0)"""
