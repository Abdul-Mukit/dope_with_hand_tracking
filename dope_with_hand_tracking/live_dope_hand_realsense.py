import numpy as np

from cuboid import *
from detector_dope import *
import yaml

import pyrealsense2 as rs

from PIL import Image
from PIL import ImageDraw
import time

# imports for yolo-hand
import cv2
import utils_orgyolo as uyolo
from darknet import Darknet


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        g_draw.ellipse(xy,
                       fill=pointColor,
                       outline=pointColor
                       )


def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius=4)
    DrawDot(points[1], pointColor=color, pointRadius=4)

    # draw x on the top
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)

# Functions for utilizing hand detection
def crop_image(image, center, newSize, plot=False):
    x_center, y_center = center
    w_new, h_new = newSize
    h_org, w_org, ch = image.shape

    x_start = x_center - w_new/2
    x_end = x_center + w_new/2

    y_start = y_center - h_new/2
    y_end = y_center + h_new/2

    if x_start < 0:
        x_end   += abs(x_start)
        x_start += abs(x_start)
    elif x_end > w_org:
        x_start -= x_end-w_org
        x_end   -= x_end-w_org-1

    if y_start < 0:
        y_end   += abs(y_start)
        y_start += abs(y_start)
    elif y_end > h_org:
        y_start -= y_end-h_org
        y_end   -= y_end-h_org-1

    img_cropped = image[y_start:y_end, x_start:x_end, :]

    if plot:
        img_plot = image.copy()
        cv2.rectangle(img_plot, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        return img_cropped, [x_start, x_end, y_start, y_end], img_plot
    else:
        return img_cropped, [x_start, x_end, y_start, y_end]

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
#######################################################
# Settings
#######################################################
exposure_val = 1000 # 166 is the default exposure value (1-10000)
use_hand_tracking = True
gamma_correction = False

hand_crop_size = [200, 200]
pose_conf_thresh = 0.5
hand_conf_thresh = 0.6
gamma_val = 2

test_width = 640
test_height = 480
fps = 30

use_cuda = True

datacfg = {'hands':   'cfg/hands.data'}

cfgfile = {'hands':   'cfg/yolo-hands.cfg',
           'cautery': 'cfg/my_config_realsense.yaml'}

weightfile = {'hands':   'backup/hands/000500.weights'}

namesfile = {'hands': 'data/hands.names'}


#######################################################
# Setting up YOLO-hand
#######################################################
model_hand = Darknet(cfgfile['hands'])
model_hand.load_weights(weightfile['hands'])
print('Loading weights from %s... Done!' % (weightfile['hands']))

if use_cuda:
    model_hand.cuda()

class_names = uyolo.load_class_names(namesfile['hands'])

#######################################################
# Setting up DOPE
#######################################################
yaml_path = cfgfile['cautery']
with open(yaml_path, 'r') as stream:
    try:
        print("Loading DOPE parameters from '{}'...".format(yaml_path))
        params = yaml.load(stream)
        print('    Parameters loaded.')
    except yaml.YAMLError as exc:
        print(exc)


    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}

    # Initialize parameters
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = pose_conf_thresh
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] = \
            ModelData(
                model,
                "backup/dope/" + params['weights'][model]
            )
        models[model].load_net_model()

        draw_colors[model] = tuple(params["draw_colors"][model])

        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )

#######################################################
# Setting up Intel RealSense Camera
#######################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, test_width, test_height, rs.format.bgr8, fps)
profile = pipeline.start(config)
# Setting exposure
s = profile.get_device().query_sensors()[1]
s.set_option(rs.option.exposure, exposure_val)

#######################################################
# Running camera and processing
#######################################################
while True:
    # Reading image from camera
    t_start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting image to RGB as DOPE expects RGB. YOLO is trained to invariant.

    # Gamma(Optional) Correction
    if gamma_correction:
        img = adjust_gamma(img, gamma=gamma_val)

    # YOLO stuff
    if use_hand_tracking:
        sized = cv2.resize(img, (model_hand.width, model_hand.height))
        bboxes = uyolo.do_detect(model_hand, sized, hand_conf_thresh, 0.4, use_cuda)
        if any(bboxes):
            center = [int(bboxes[0][0] * test_width), int(bboxes[0][1] * test_height)]
            img_hand_cropped, crop_box = crop_image(img, center, hand_crop_size)
            img_detection = img_hand_cropped
            # print(crop_box[3])
        else:
            img_detection = img
    else:
        img_detection = img

    # DOPE pose detection

    for m in models:
        # Detect object
        t_start_dope = time.time()
        results = ObjectDetector.detect_object_in_image(
            models[m].net,
            pnp_solvers[m],
            img_detection,
            config_detect
        )
        t_end_dope = time.time()
        # Overlay cube on image. Copy and draw image
        img_copy = img_detection.copy()
        img_draw = Image.fromarray(img_copy)
        g_draw = ImageDraw.Draw(img_draw)

        for i_r, result in enumerate(results):
            if result["location"] is None:
                continue
            loc = result["location"]
            ori = result["quaternion"]

            # Draw the cube
            if None not in result['projected_points']:
                points2d = []
                for pair in result['projected_points']:
                    points2d.append(tuple(pair))
                DrawCube(points2d, draw_colors[m])

    img_draw = np.array(img_draw)# Converstion to cv2 image (numpy array)

    # Stitching the cropped image if hand detected
    if use_hand_tracking and any(bboxes):
        img[crop_box[2]:crop_box[3], crop_box[0]:crop_box[1], :] = img_draw
        img_draw = img
        cv2.rectangle(img_draw, (crop_box[0],crop_box[2]), (crop_box[1],crop_box[3]), color=(0,255,0))





    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)


    cv2.imshow('Open_cv_image', img_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t_end = time.time()
    print('Overall FPS: {}, DOPE fps: {}'.format(1/(t_end-t_start), 1/(t_end_dope-t_start_dope)))









