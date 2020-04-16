import ntpath
import os

vid_path = r"C:\Users\Ghost\Downloads\Amm\Azure_Kinect_Captures_1\outputAutoExposure.mkv"
vid_f_name = ntpath.basename(vid_path)
vid_f_name = os.path.splitext(vid_f_name)[0]


print(vid_f_name)