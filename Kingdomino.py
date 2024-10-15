import numpy as np
import cv2
import glob

# Import scorecard dataset
from crowns_array import scorecard

"""
Get colors of each tile from Kingdomino game
"""
def Kingdomino_colorEvaluate(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img, 50, 75) 

    # Apply thresholding to get colors of tiles
    _, img_blue = cv2.threshold(img[:,:,0], 100, 255, cv2.THRESH_BINARY)
    _, img_yellow = cv2.threshold(img[:,:,2], 175, 255, cv2.THRESH_BINARY) # try 160 # ALT: derive from max value of red
    _, img_forest = cv2.threshold(img[:,:,2] - img[:,:,1], 225, 255, cv2.THRESH_BINARY)
    _, img_greenR = cv2.threshold(img[:,:,2], 127, 255, cv2.THRESH_BINARY) # 127
    _, img_greenG = cv2.threshold(img[:,:,1], 127, 255, cv2.THRESH_BINARY)
    img_green = img_greenG - img_greenR
    _, img_mines = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    _, img_desert = cv2.threshold(img[:,:,0], 30, 255, cv2.THRESH_BINARY)

    castleMatch = np.empty(25)
    # For every tile in presumed 5x5 grid on 500x500 image
    for tile in range(25):
        tileRow = (tile//5)*100
        tileColumn = (tile%5)*100

        # A 100x100 image has 10000 pixels thresholded to values of either 0 or 255
        # Get a value spanning [0; 1] to determine how well a tile matches a color pattern
        yellowMatch = np.sum(img_yellow[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 
        blueMatch = np.sum(img_blue[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255)
        forestMatch = np.sum(img_forest[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 
        greenMatch = np.sum(img_green[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 
        minesMatch = np.sum(img_mines[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 
        desertMatch = np.sum(img_desert[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 
        castleMatch[tile] = np.sum(img_canny[tileRow:tileRow+100, tileColumn:tileColumn+100]) / (10000 * 255) 

        # PRIORITY LIST:
        # 1. Yellow - Is isolated completely using threshold
        # 2. Blue - Is isolated completely using threshold
        # 3. forest - Is isolated completely using threshold (Castle tile is discernable in some instances)
        # 4. green - Is isolated completely using threshold (Castle tile is discernable in some instances)
        # 5. mines - Contains forests tiles
        # 6. desert - Contains blue tiles AND CASTLE!!!
        if yellowMatch > 0.25:
            score_colors[tile] = 'Y'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (0, 255, 255), -1)
        elif blueMatch > 0.5:
            score_colors[tile] = 'B'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (255, 0, 0), -1)
        elif forestMatch > 0.5:
            score_colors[tile] = 'F'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (0, 127, 0), -1)
        elif greenMatch > 0.25:
            score_colors[tile] = 'G'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (0, 255, 0), -1)
        elif minesMatch > 0.5:
            score_colors[tile] = 'M'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (0, 0, 0), -1)
        elif desertMatch > 0.5:
            score_colors[tile] = 'D'
            cv2.rectangle(img, (tileColumn+40, tileRow+40), (tileColumn+60, tileRow+60), (127, 127, 127), -1)


    # Get Castle tile using canny edge detection
    # Method is not 100% perfect, castles placed on top of tile causes errors (sometimes)
    castle_tile = np.argmax(castleMatch)
    score_colors[castle_tile] = 'C'
    cv2.rectangle(img, ((castle_tile%5)*100+40, (castle_tile//5)*100+40), ((castle_tile%5)*100+60, (castle_tile//5)*100+60), (255, 255, 255), -1)


"""
Get crowns of tiles in Kingdomino game.
"""
def Kingdomino_crownEvaluate(img, template_crown):
    img_template = np.zeros_like(img, dtype=np.uint8)
    for y, row in enumerate(img):
        for x, pixel in enumerate(row): 
            pixel = [0, 0, 0]

            # If each (Red)(Green)(Blue) RGB-pixel has 8-bit value above 80
            if (img[y][x][:] > 80).all():
                pixel = [255, 255, 255]
        
            img_template[y, x] = pixel

    img_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    pts = []

    for i in range(4):
        # Use template matching to find crowns in each orientation
        patternMatch = cv2.matchTemplate(img_template, template_crown, cv2.TM_CCOEFF_NORMED)
        _, img_crowns = cv2.threshold(patternMatch, 0.5, 255, cv2.THRESH_BINARY)

        # Rotate template 90 degrees
        template_crown = cv2.rotate(template_crown, cv2.ROTATE_90_CLOCKWISE)
        w, h = template_crown.shape

        # Get coordinates of each pixel with HIGH value
        locations = np.where(img_crowns >= 0.5)

        for pt in zip(*locations[::-1]):  # Switch columns and rows
            pts.append(pt)

            # Draw rectangle on the original image
            cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

            # Each point is claustrophobic - remove points to near to prior registered points
            for i in range(len(pts)-1):
                if pt[0] > pts[i][0] - w//2 and pt[0] < pts[i][0] + w//2 and pt[1] > pts[i][1] - h//2 and pt[1] < pts[i][1] + h//2:
                    pts.pop() # Get rid of point
                    break
            
    # Add each point - pixel - from pts[] as an additional crown to specified tile
    for i in range(len(pts)):
        # Get center point of given point
        pt_center = [(pts[i][0]+w//2), (pts[i][1]+h//2)]

        pt_score_index = (pt_center[0]//100) + (pt_center[1]//100)*5
        tem = ord(score_crowns[pt_score_index])
        score_crowns[pt_score_index] = chr(tem+1)

"""
A Score calculate for Kingdomino using:
-> A char[25] array of colors
-> A char[25] array of test crowns

Using the following logic:
'Y' = yellow; 
'F' = forest; 
'B' = blue; 
'D' = desert; 
'G' = green; 
'M' = mine
'C' = castle
"""
def kingdomino_scoreCalculator(game_colors, game_crowns):
    score = 0

    # A 5x5 grid always contains 25 tiles. 
    # column = i // 5 # <- i divided by 5 rounded down
    # row = i % 5 # <- i modulus 5
    for i in range(25):
        # Only deal with blocks with crowns
        if game_crowns[i] == '' or game_crowns[i] == '0':
            continue
        
        # Initialize a burn queue and store adjacent colors and crowns to i
        burn_queue = [i]
        adjacent_colors = []
        adjacent_crowns = 0

        # While burn queue has tiles to burn
        while len(burn_queue) > 0:
            #print(len(burn_queue))
            # Allocate most recent entry to burn queue
            current = burn_queue.pop()

            # Pass duplicate tiles
            if current in burn_queue:
                continue

            # Append entry color & add entry crowns
            adjacent_colors.append(game_colors[current])
            if game_crowns[current] != '':
                adjacent_crowns += int(game_crowns[current])

            # Check if neighboring tiles have same color as current entry
            if game_colors[current] == game_colors[current-5] and current-5 > 0 and game_colors[current-5] != '':
                burn_queue.append(current-5)
            if game_colors[current] == game_colors[current-1] and (current-1) // 5 == current // 5 and game_colors[current-1] != '':
                burn_queue.append(current-1)
            if current+1 < 25:
                if game_colors[current] == game_colors[current+1] and (current+1) // 5 == current // 5 and game_colors[current+1] != '':
                    burn_queue.append(current+1)
            if current+5 < 25:
                if game_colors[current] == game_colors[current+5] and game_colors[current+5] != '':
                    burn_queue.append(current+5)

            # Remove aka. 'burn' entry colors and crowns of tile
            game_colors[current] = ''
            game_crowns[current] = ''

        # Add score acquired by burn queue to total score
        score += adjacent_crowns * len(adjacent_colors)

    return score

# Extract files from each of the folders containing dominos, castles, and games
game_paths = sorted(glob.glob("Cropped and perspective corrected boards/*.jpg"))

template_crown = cv2.imread("crown.jpg", cv2.IMREAD_GRAYSCALE)

# [file, [x, y, RGB]]
game = []
print("Loading dataset of 500x500 Kingdomino games...")
for i in range(len(game_paths)):
    print(i)
    temp = cv2.imread(game_paths[i])
    game.append( cv2.resize(temp, (500, 500)) )


# Evaluate each game one at a time
for i in range(len(game)): # len(game)
    # np.copy() is used to store duplicates of numpy array in memory
    img_original = np.copy(game[i])
    img = np.copy(game[i])

    # Reset 5x5 grid colors and crowns
    score_colors = [''] * 25
    score_crowns = ['0'] * 25

    # Template evaluate all dominos with mines
    Kingdomino_colorEvaluate(img)
    Kingdomino_crownEvaluate(img, template_crown)

    print(score_colors)
    print(score_crowns)

    # Acquire total score using evaluated colors and crowns.
    score = kingdomino_scoreCalculator(score_colors, score_crowns)

    print("Kingdomino score of game \t" + str(i+1) + ":")
    print("Calculated score: \t" + str(score))
    print("Scorecard: \t" + str(scorecard[i]))

    if score != scorecard[i]:
        cv2.imshow("Original", img_original)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
