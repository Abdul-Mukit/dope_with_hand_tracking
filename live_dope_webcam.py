import numpy as np

from cuboid import *
from detector_dope import *
import yaml

import pyrealsense2 as rs

from PIL import Image
from PIL import ImageDraw


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


# Settings
config_name = "my_config_webcam.yaml"
exposure_val = 166


yaml_path = 'cfg/{}'.format(config_name)
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
    config_detect.threshold = 0.5
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

    cap = cv2.VideoCapture(0)

    while True:
        # Reading image from camera
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            continue

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        g_draw = ImageDraw.Draw(im)

        for m in models:
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                models[m].net,
                pnp_solvers[m],
                img,
                config_detect
            )

            # Overlay cube on image
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

        open_cv_image = np.array(im)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)


        cv2.imshow('Open_cv_image', open_cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break









