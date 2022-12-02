from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np

def coord_to_pixel(pointLon, pointLat, im, x_offset=0, y_offset=0):
    imageWidth, imageHeight = im.size
    ImageExtentLeft = -87.7478434782057
    ImageExtentRight = -87.56321752669828
    ImageExtentTop = 42.032452016596274
    ImageExtentBottom = 41.641470583774385
    
    x = imageWidth * ( pointLon - ImageExtentLeft ) / (ImageExtentRight - ImageExtentLeft)
    y = imageHeight * ( 1 - ( pointLat - ImageExtentBottom) / (ImageExtentTop - ImageExtentBottom))
    return x+x_offset, y+y_offset


im_path = "data\chicago_night_processed.png"
geo_coord_center = -87.66552272954672, 41.89016056612478

im = Image.open(im_path)
size = im.size
pic_x = size[0]/4
pic_y = size[1]/4

plt.imshow(im)
plt.scatter(pic_x, pic_y, c='red', s=20)


x, y = coord_to_pixel(geo_coord_center[0], geo_coord_center[1], im)
# print(pic_x-x, pic_y-y)
plt.scatter(x, y, c='blue', s=20, marker='x')

# united center
# x, y = coord_to_pixel(-87.67421094151734, 41.88068289515608, im)
# plt.scatter(x, y, c='green', s=20) 



plt.show()
