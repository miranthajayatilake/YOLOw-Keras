# import the needed modules
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval


#Provide the name of the image that you saved in the images folder to be fed through the network
input_image_name = "test45.jpg"

#Obtaining the dimensions of the input image
input_image = Image.open("images/" + input_image_name)
width, height = input_image.size
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)

#Assign the shape of the input image to image_shapr variable
image_shape = (height, width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
image, image_data = preprocess_image("images/" + input_image_name, model_image_size = (608, 608))

#Run the session
out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


#Print the results
print('Found {} boxes for {}'.format(len(out_boxes), input_image_name))
#Produce the colors for the bounding boxs
colors = generate_colors(class_names)
#Draw the bounding boxes
draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
#Apply the predicted bounding boxes to the image and save it
image.save(os.path.join("out", input_image_name), quality=90)
output_image = scipy.misc.imread(os.path.join("out", input_image_name))
imshow(output_image)
