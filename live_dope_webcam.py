import numpy as np

from cuboid import *
from detector_dope import *
import yaml

#import pyrealsense2 as rs

from PIL import Image
from PIL import ImageDraw


### Code to visualize the neural network output

def DrawLine(point1: object, point2: object, lineColor: object, lineWidth: object) -> object:
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
    DrawLine(points[0], points[1], color, 3)
    DrawLine(points[1], points[2], color, 3)
    DrawLine(points[3], points[2], color, 3)
    DrawLine(points[3], points[0], color, 3)

    # draw back
    DrawLine(points[4], points[5], color, 1)
    DrawLine(points[6], points[5], color, 1)
    DrawLine(points[6], points[7], color, 1)
    DrawLine(points[4], points[7], color, 1)

    # draw sides
    DrawLine(points[0], points[4], color, 2)
    DrawLine(points[7], points[3], color, 2)
    DrawLine(points[5], points[1], color, 2)
    DrawLine(points[2], points[6], color, 2)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius=4)
    DrawDot(points[1], pointColor=color, pointRadius=4)

    # draw x on the top
    DrawLine(points[0], points[5], color, 1)
    DrawLine(points[1], points[4], color, 1)


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # Test Resize Dimensions
        width = 640 # can get params from my_config_webcam.yaml ##Uncomment to override
        height = 480
        dim = (width, height)

        # resize image - Custom resolution to override native resolution:
        # Optimal detection distance is proportional to camera resolution
        # 640x480: Optimal distance 45-50cm 1280x720: Optimal distance 90cm with native pixel spacing

        #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) ##Uncomment for custom resolution

        #crop to center square
        # 400: This resolution was used in training and is optimal accuracy of DOPE pose detections
        # 224: Is the input resolution of VGG19 and gives the fastest fps, jittery pose detections due to screen-door effect
        # Correction factor for location matrix results: crop/height (todo: is quaternion correction needed?)

        crop = 400
        margin_y = (height-crop)//2
        margin_x = (width-crop)//2
        #img = img[margin_y:margin_y+crop, margin_x:margin_x+crop] #numpy img format(y,x)


        # Copy and draw image (?to PIL)
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
                print("location ", loc, "quaternion ", ori)

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









