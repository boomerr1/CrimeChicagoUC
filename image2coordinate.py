from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

def coord_to_pixel(pointLon, pointLat, im, x_offset=4.204476361908064, y_offset=35.22609355416489):
    imageWidth, imageHeight = im.size
    ImageExtentLeft = -87.7478434782057
    ImageExtentRight = -87.56321752669828
    ImageExtentTop = 42.032452016596274
    ImageExtentBottom = 41.641470583774385
    
    x = imageWidth * ( pointLon - ImageExtentLeft ) / (ImageExtentRight - ImageExtentLeft)
    y = imageHeight * ( 1 - ( pointLat - ImageExtentBottom) / (ImageExtentTop - ImageExtentBottom))
    return x+x_offset, y+y_offset


im_path = "chicago_night_processed.png"
geo_coord = -87.60637807060172, 41.86662280291246

im = Image.open(im_path)
pic_x = 1116
pic_y = 1778

plt.imshow(im)
plt.scatter(pic_x, pic_y, c='red', s=20)


x, y = coord_to_pixel(geo_coord[0], geo_coord[1], im)
plt.scatter(x, y, c='blue', s=20) 

# united center
x, y = coord_to_pixel(-87.67421094151734, 41.88068289515608, im)
plt.scatter(x, y, c='green', s=20) 



plt.show()
