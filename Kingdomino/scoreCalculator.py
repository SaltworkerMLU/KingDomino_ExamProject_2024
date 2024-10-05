import cv2

# colors and crowns of game #2
test_colors = ['Y', 'D', 'Y', 'M', 'D',
               'M', 'D', 'Y', 'Y', 'F',
               'M', 'D', 'C', 'Y', 'Y',
               'M', 'M', 'G', 'G', 'Y',
               'Y', 'Y', 'D', 'D', 'G'] # 'Y', 'Y', 'D', ...

test_crowns = ['0', '0', '0', '2', '0',
               '1', '1', '0', '0', '0',
               '2', '0', '0', '0', '0',
               '2', '3', '0', '0', '1',
               '0', '0', '0', '0', '0']

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
            # Allocate most recent entry to burn queue
            current = burn_queue.pop()

            # Append entry color & add entry crowns
            adjacent_colors.append(game_colors[current])
            adjacent_crowns += int(game_crowns[current])

            # Check if neighboring tiles have same color as current entry
            if game_colors[current] == game_colors[current-5] and current-5 > 0:
                burn_queue.append(current-5)
            if game_colors[current] == game_colors[current-1] and (current-1) // 5 == current // 5:
                burn_queue.append(current-1)
            if game_colors[current] == game_colors[current+1] and (current+1) // 5 == current // 5:
                burn_queue.append(current+1)
            if game_colors[current] == game_colors[current+5] and current-5 < 25:
                burn_queue.append(current+5)

            # Remove aka. 'burn' entry colors and crowns of tile
            game_colors[current] = ''
            game_crowns[current] = ''

        # Add score acquired by burn queue to total score
        score += adjacent_crowns * len(adjacent_colors)

    return score

def patternRecognizer(image, kernel, thresh=1.0):
    patternMatch = cv2.matchTemplate(image, kernel, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(patternMatch)
    patternMatch = cv2.cvtColor(patternMatch, cv2.IMREAD_GRAYSCALE)
    
    # Always go for the best template match - the highest value
    if thresh < max_val:
        print(max_val)
        thresh = max_val - 0.01

    # 8-bit provided in percentage
    ret, result = cv2.threshold(patternMatch, thresh, 255, cv2.THRESH_BINARY)

    return result # patternMatch # 