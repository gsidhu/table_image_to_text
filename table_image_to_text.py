import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import argparse
import logging as log
import easyocr

reader = easyocr.Reader(['en'])

WORKING_DIRECTORY = os.getcwd()
IMAGE_LOCATION = ''


#####################################
# functions for identifying squares
# borrowed from OpenCV Python samples: https://github.com/opencv/opencv/blob/master/samples/python/squares.py
#####################################
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img, threshold):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < threshold:
                        squares.append(cnt)
    return squares

###################################
# find the squares in given image #
###################################
def find_squares_in_image():
    global IMAGE_LOCATION
    image = cv.imread(IMAGE_LOCATION)
    squares = find_squares(image, 0.1)
    cv.drawContours(image, squares, -1, (0,255,0),3)
    cv.imshow('squares', image)

    # show the image
    # plt.imshow(image)
    # plt.show()
    return squares

#####################################
# functions for identifying squares #
#####################################

def find_top_cords(squares, threshold=10):
    # find the top x coords of all the columns; where do they begin
    top_x = []
    for i in squares:
        top_x.append(i[0][0])
    top_x = np.unique(top_x)

    x_bins = [top_x[0]]
    for i in top_x:
        # get the optimal bin value
        # distance b/w x_cords of two cols should be at least 50 units
        if i - x_bins[-1] > threshold:
            x_bins.append(i)

    # find the top y coords of all the columns; where do they begin
    top_y = []
    for i in squares:
        top_y.append(i[0][1])
    top_y = np.unique(top_y)

    y_bins = [top_y[0]]
    for i in top_y:
        # get the optimal bin value
        # distance b/w x_cords of two cols should be at least 50 units
        if i - y_bins[-1] > threshold:
            y_bins.append(i)

    # make pairs of top coords
    top_cords = []
    for i in x_bins[1:-1]:
        for j in y_bins[1:]:
            top_cords.append((i,j))

    return top_cords

def find_bottom_cords(squares, threshold=10):
    # find the bottom x coords of all the columns; where do they end
    top_x = []
    for i in squares:
        top_x.append(i[2][0])
    top_x = np.unique(top_x)

    x_bins = [top_x[0]]
    for i in top_x:
        # get the optimal bin value
        # distance b/w x_cords of two cols should be at least 50 units
        if i - x_bins[-1] > threshold:
            x_bins.append(i)

    # find the bottom y coords of all the columns; where do they end
    top_y = []
    for i in squares:
        top_y.append(i[2][1])
    top_y = np.unique(top_y)

    y_bins = [top_y[0]]
    for i in top_y:
        # get the optimal bin value
        # distance b/w x_cords of two cols should be at least 50 units
        if i - y_bins[-1] > threshold:
            y_bins.append(i)

    # make pairs of top coords
    bottom_cords = []
    for i in x_bins[2:-1]:
        if i != 0:
            for j in y_bins[:-1]:
                if j != 0:
                    bottom_cords.append((i,j))
    
    # update ncols and nrows
    global nrows
    global ncols
    nrows = len(y_bins) - 1 # remove the first y_bin cuz it's at the edge
    ncols = len(x_bins) - 2 # remove first and last x_bins cuz they're at edges

    print("Found table with %d rows and %d columns" % (nrows, ncols))
    return bottom_cords

def view_identified_crops(top_cords, bottom_cords):
    global IMAGE_LOCATION
    im = plt.imread(IMAGE_LOCATION)
    implot = plt.imshow(im)

    for i in top_cords:
        plt.scatter([i[0]], [i[1]], marker='x', c='r')
    for i in bottom_cords:
        plt.scatter([i[0]], [i[1]], marker='o', c='g')
    plt.show()

def crop_image(squares, index, padding=[0,0,0,0]):
    global IMAGE_LOCATION
    global WORKING_DIRECTORY
    from PIL import Image
  
    # Open image in RGB mode
    im = Image.open(IMAGE_LOCATION)

    # Size of the image in pixels (size of orginal image)
    # (This is not mandatory)
    width, height = im.size
    
    # padding order: top, right, bottom, left
    # Setting the points for cropped image
    left, top = squares[0][0]-padding[3], squares[0][1]-padding[0]
    right, bottom = squares[1][0]+padding[1], squares[1][1]+padding[2]
    
    # Cropped image of above dimension
    # (It will not change orginal image)
    im1 = im.crop((left, top, right, bottom))

    filename = WORKING_DIRECTORY + '/temp_images/' + str(index) + '.png'
    im1.save(filename)


# create crops
def create_crops(top_cords, bottom_cords):
    all_cords = list(zip(top_cords, bottom_cords))

    # create temp directory to store cropped images
    if not os.path.exists('temp_images'):
        os.mkdir('temp_images')

    for i in range(len(all_cords)):
        crop_image(all_cords[i], index=i, padding=[2,2,2,2])

    # create rows with all columns to create html/md output
    square_rows = [] # contains square coordinates
    index_rows = [] # contains index of the square
    text_rows = [] # contains parsed text of the square
    for i in range(nrows):
        square_rows.append([])
        index_rows.append([])
        text_rows.append([])
        
    for i in range(len(all_cords)):
        rownum = i%nrows
        square_rows[rownum].append(all_cords[i])
        index_rows[rownum].append(i)
        text_rows[rownum].append(i)

    return index_rows, text_rows

def html_output(text_rows, headers=True):
    # html output
    html = '''<html>
            <head>
                <style>
                td { border: 1px solid black }
                </style>
            </head>
            <body>
            <table>'''
    colnum = 0

    for i in range(len(text_rows)):
        row = text_rows[i]
        html_row = ''
        for cell in row:
            result = str(cell)
            if i == 0 and headers:
                html_cell = '<th>' + result + '</th>'
            else:
                html_cell = '<td>' + result + '</td>'
            html_row += html_cell
        html_row = '<tr>' + html_row + '</tr>'
        html += html_row

    html += '</table></body></html>'
    filename = WORKING_DIRECTORY + '/output.html'
    with open(filename, 'w+') as f:
        f.write(html)

def md_output(text_rows, headers=True):
    # markdown output
    md = ''
    colnum = 0
    if not headers:
        md += '|' * (ncols)
        md += '|\n'
        md += '|--' * (ncols) 
        md += '|\n'
    for i in range(len(text_rows)):
        row = text_rows[i]
        md_row = '|'
        for cell in row:
            result = str(cell)
            md_row += result + '|'
        md += md_row + '\n'
        if i == 0 and headers:
            md += '|--' * (ncols) 
            md += '|\n'
    filename = WORKING_DIRECTORY + '/output.md'
    with open(filename, 'w+') as f:
        f.write(md)

# Run OCR on the cropped images
def run_ocr(index_rows, text_rows, output='html', headers=True):
    global reader

    for i in range(len(index_rows)):
        row = index_rows[i]
        for j in range(len(row)):
            cell = row[j]
            crop = WORKING_DIRECTORY + '/temp_images/' + str(cell) + '.png'
            result = reader.readtext(crop, detail=0)
            result = ' '.join(result)
            text_rows[i][j] = result

    # delete temporary files
    import shutil 
    shutil.rmtree('temp_images')

    # generate output
    if output == 'html':
        html_output(text_rows, headers)
    elif output == 'md':
        md_output(text_rows, headers)
    elif output == 'both':
        html_output(text_rows, headers)
        md_output(text_rows, headers)
    
# Run the whole script
def get_text_from_table(filepath, output='html', headers=True, view_cells=False):
    global IMAGE_LOCATION
    IMAGE_LOCATION = filepath

    # find squares
    squares = find_squares_in_image()
    print("found squares")
    # find coords
    top_cords = find_top_cords(squares, threshold=50)
    bottom_cords = find_bottom_cords(squares, threshold=50)
    print("found coordinates")
    # view identified crops
    if view_cells == True:
        view_identified_crops(top_cords, bottom_cords)
    # create crops
    index_rows, text_rows = create_crops(top_cords, bottom_cords)
    print("created crops")
    # run OCR and output
    run_ocr(index_rows, text_rows, output, headers)
    print("OCR complete")

########################
# Command line utility #
########################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reads text from a table in an image and outputs in html or markdown.")

    parser.add_argument("-f", "--filepath", type=str, help="path to the image file")
    parser.add_argument("-o", "--output", type=str, default='html', help="output type: html or md")
    parser.add_argument("-H" , "--headers", type=str, default='y', help="are there headers in the table: y/N")
    parser.add_argument("-c" , "--cells", type=str, default='n', help="show detected cells in table: Y/n")

    # Verbosity and Debugging
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=log.DEBUG,
        default=log.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", const=log.INFO,
    )

    args = parser.parse_args()

    if args.headers == 'y':
        args.headers = True
    elif args.headers == 'n':
        args.headers = False
    
    if args.cells == 'y':
        args.cells = True
    elif args.cells == 'n':
        args.cells = False

    print("Reading file: %s" % args.filepath)
    print("Generating %s output with headers set to %s" % (args.output, args.headers))
    print("Show identified cells set to %s" % args.cells)

    get_text_from_table(args.filepath, args.output, args.headers, args.cells)
