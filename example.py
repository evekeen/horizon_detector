import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

image_dir = './images/'

data = pd.read_csv('metadata.csv', header=None)
images = data[0].tolist()
left_x = data[1].tolist()
left_y = data[2].tolist()
right_x = data[3].tolist()
right_y = data[4].tolist()

for ix in range(len(images)):
    img_path = os.path.join(image_dir, images[ix])
    im = Image.open(img_path)
    width, height = im.size
    print(f"Image dimensions: {width}x{height}")
    print(f"Line coordinates: ({left_x[ix]}, {left_y[ix]}) to ({right_x[ix]}, {right_y[ix]})")
    
    # Create figure with the exact pixel size of the image
    plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Calculate extent to match MATLAB's coordinate system
    # In MATLAB: 'XData', [1 sz(2)] - (sz(2)+1)/2, 'YData', [sz(1) 1] - (sz(1)+1)/2
    x_left = 1 - (width+1)/2
    x_right = width - (width+1)/2
    y_top = height - (height+1)/2
    y_bottom = 1 - (height+1)/2
    
    # Display the image with the same coordinate system as MATLAB
    plt.imshow(im, extent=[x_left, x_right, y_bottom, y_top])
    plt.axis('off')
    
    # Plot the line with the same coordinates as in MATLAB
    plt.plot([left_x[ix], right_x[ix]], [left_y[ix], right_y[ix]], 'g', linewidth=3)
    
    plt.tight_layout(pad=0)
    plt.show(block=False)
    plt.pause(0.1)
    input("Press Enter to continue...")
    plt.close()
