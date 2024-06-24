import cv2
import numpy as np

def _default_camera_params():

    fx = 1093.2768
    fy = 1093.2768
    cx = 964.989
    cy = 569.276

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K

def _default_distortion_params():

    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distortion = np.array([k1, k2, p1, p2])

    return distortion

def _default_stereo_baseline():

    # baseline = -0.11972
    baseline = 0.13
    t = np.array([baseline,0,0])

    return t

def _default_stereo_rotation():

    R = np.eye(3)
    return R

def sm_frame2pcl(imgL, imgR, disparity, K, D, R, t,visualize=False):

    if visualize:
        cv2.imshow('DisparityMap', cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        cv2.waitKey()
        cv2.destroyAllWindows()

    # Generate 3D point cloud from disparity map and camera parameters
    h, w = imgL.shape[:2]
    Q = cv2.stereoRectify(K,D,K,D,[w,h],R,t)[4]
    points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # Remove points with invalid depth values
    mask = disparity > disparity.min()

    # Filter points with valid depth
    points = points[mask]
    colors = imgL[mask]

    return points, colors

def main(imgL, imgR, disparity):
    K = _default_camera_params()
    D = _default_distortion_params()
    R = _default_stereo_rotation()
    t = _default_stereo_baseline()
    # imgL = cv2.imread('/path/to/left/image')
    # imgR = cv2.imread('/path/to/right/image')
    # disparity = cv2.imread('/path/to/disparity/image')
    points, colors = sm_frame2pcl(imgL, imgR, disparity, K, D, R, t, visualize=False)
    return points, colors

if __name__=="__main__":
    K = _default_camera_params()
    D = _default_distortion_params()
    R = _default_stereo_rotation()
    t = _default_stereo_baseline()
    imgL = cv2.imread('/path/to/left/image')
    imgR = cv2.imread('/path/to/right/image')
    disparity = cv2.imread('/path/to/disparity/image')
    points, colors = sm_frame2pcl(imgL, imgR, disparity, K, D, R, t, visualize=False)