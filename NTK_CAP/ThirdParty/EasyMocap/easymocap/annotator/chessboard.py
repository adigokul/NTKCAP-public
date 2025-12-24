'''
  @ Date: 2021-04-13 16:14:36
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-25 20:56:26
  @ FilePath: /EasyMocapRelease/easymocap/annotator/chessboard.py
'''
import numpy as np
import cv2
from func_timeout import func_set_timeout
import os
import shutil

# Try to import YOLO, but make it optional
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

from easymocap.annotator.GUI_detectchessboard import Manual_GUI_detection
def getChessboard3d(pattern, gridSize, axis='xy'):
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points = object_points * gridSize
    if axis == 'zx':
        object_points = object_points[:, [1, 2, 0]]
    return object_points

colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]

def get_lines_chessboard(pattern=(9, 6)):
    w, h = pattern[0], pattern[1]
    lines = []
    lines_cols = []
    for i in range(w*h-1):
        lines.append([i, i+1])
        lines_cols.append(colors_chessboard_bar[(i//w)%len(colors_chessboard_bar)])
    return lines, lines_cols

def _findChessboardCorners(img, pattern,imgname, debug):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #retval, corners = cv2.findChessboardCorners(img, pattern, 
        #flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    
    retval, corners = cv2.findChessboardCornersSB(img,pattern,cv2.CALIB_CB_NORMALIZE_IMAGE)

    
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners

def _findChessboardCornersAdapt(img, pattern,imgname, debug):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2)
    # cv2.imshow('Adaptive Threshold', img)
    # cv2.waitKey(0)
    return _findChessboardCorners(img, pattern,imgname, debug)
def _findChessboardCornersYOLO(img, pattern,imgname ,debug):
    """YOLO-based chessboard detection - requires yolo_model_v1.pt"""
    # Check if YOLO is available and model exists
    if not YOLO_AVAILABLE:
        return False, None
    
    model_path = 'yolo_model_v1.pt'
    if not os.path.exists(model_path):
        # Try to find model in common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_paths = [
            os.path.join(script_dir, model_path),
            os.path.join(script_dir, '..', '..', model_path),
            os.path.join(os.getcwd(), model_path),
        ]
        model_found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                model_found = True
                break
        if not model_found:
            # Model not found, skip YOLO detection
            return False, None
    
    try:
        model_trained = YOLO(model_path)   
        img = cv2.imread(imgname)
        result_ex = model_trained.predict(source =imgname, save=False, conf=0.5, max_det=1)
        if len(result_ex[0].boxes.data) == 0:
            return False, None
        x_min, y_min, x_max, y_max, _, _ = result_ex[0].boxes.data[0]       
        img[:int(y_min), :] = np.array([0, 255, 0])
        img[int(y_max):, :] = np.array([0, 255, 0])
        img[:, :int(x_min)] = np.array([0, 255, 0])
        img[:, int(x_max):] = np.array([0, 255, 0])
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY, 21, 2)
        return _findChessboardCorners(img, pattern,imgname, debug)
    except Exception as e:
        # YOLO detection failed, return False to try other methods
        return False, None

def validate_corners_grid(corners, pattern):
    """
    Validate that detected corners form a proper grid pattern.
    
    This is a lightweight check - OpenCV's findChessboardCornersSB is reliable,
    so we only reject obviously wrong detections (e.g., random noise).
    
    Returns True if valid, False if corners appear completely scrambled.
    """
    try:
        rows, cols = pattern
        total_corners = rows * cols
        
        # Basic sanity check: correct number of corners
        if len(corners) != total_corners:
            return False
        
        corners_2d = corners.reshape(rows, cols, 2)
        
        # Check that corners span a reasonable area (not all clustered in one spot)
        all_x = corners_2d[:, :, 0].flatten()
        all_y = corners_2d[:, :, 1].flatten()
        
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        
        # If all corners are within 50 pixels, something is wrong
        if x_range < 50 and y_range < 50:
            return False
        
        # Trust OpenCV's detection for everything else
        return True
        
    except Exception as e:
        # Don't block on validation error
        return True

@func_set_timeout(5000)
def findChessboardCorners(img, annots, pattern,imgname, debug=False):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    methods_tried = []
    for func_name, func in [("_findChessboardCorners", _findChessboardCorners),
                           ("_findChessboardCornersAdapt", _findChessboardCornersAdapt),
                           ("_findChessboardCornersYOLO", _findChessboardCornersYOLO)]:
        ret, corners = func(gray, pattern, imgname, debug)
        methods_tried.append(func_name)
        if ret:
            # Validate corners form a proper grid
            if not validate_corners_grid(corners, pattern):
                print(f"[WARNING] Detected corners for {imgname} using {func_name} appear scrambled, trying next method...")
                ret = False
                continue
            break
    else:
        # All methods failed
        print(f"[ERROR] Chessboard detection failed for {imgname} after trying: {', '.join(methods_tried)}")
        print(f"        This usually means the chessboard is not visible, poorly lit, or has glare.")
        print(f"        Try: Better lighting, remove glare/reflections, ensure full chessboard is in frame.")
        return None

    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show
def findChessboardCorners_manual(img, annots, pattern,imgname, debug=False):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    # Find the chess board corners
    corners = Manual_GUI_detection(img)
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners,True)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show
def create_chessboard(path, keypoints3d, out='annots'):
    from tqdm import tqdm
    from os.path import join
    from .file_utils import getFileList, save_json, read_json
    import os
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(join(path, 'images'), ext='.jpg', max=1)
    imgnames = [join('images', i) for i in imgnames]
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', out).replace('.jpg', '.json')
        annname = join(path, annname)
        if not os.path.exists(annname):
            save_json(annname, template)
        elif True:
            annots = read_json(annname)
            annots['keypoints3d'] = template['keypoints3d']
            save_json(annname, annots)


def detect_charuco(image, aruco_type, long, short, squareLength, aruco_len):
    ARUCO_DICT = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
    }
    # 创建ChArUco标定板
    dictionary = cv2.aruco.getPredefinedDictionary(dict=ARUCO_DICT[aruco_type])
    board = cv2.aruco.CharucoBoard_create(
        squaresY=long,
        squaresX=short,
        squareLength=squareLength,
        markerLength=aruco_len,
        dictionary=dictionary,
    )
    corners = board.chessboardCorners
    # ATTN: exchange the XY
    corners3d = corners[:, [1, 0, 2]]
    keypoints2d = np.zeros_like(corners3d)
    # 查找标志块的左上角点
    corners, ids, _ = cv2.aruco.detectMarkers(
        image=image, dictionary=dictionary, parameters=None
    )
    # 棋盘格黑白块内角点
    if ids is not None:
        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=image, board=board
        )
        if retval:
            ids = charucoIds[:, 0]
            pts = charucoCorners[:, 0]
            keypoints2d[ids, :2] = pts
            keypoints2d[ids, 2] = 1.
    else:
        retval = False
    return retval, keypoints2d, corners3d

class CharucoBoard:
    def __init__(self, long, short, squareLength, aruco_len, aruco_type) -> None:    
        '''
            short,long 分别表示短边、长边的格子数.
            squareLength,aruco_len 分别表示棋盘格的边长与aruco的边长.
            aruco_type 表示Aruco的类型 4X4表示aruco中的白色格子是4x4的 _50表示aruco字典中有多少种aruco.
        '''
        # 定义现有的Aruco类型
        self.ARUCO_DICT = {
            "4X4_50": cv2.aruco.DICT_4X4_50,
            "4X4_100": cv2.aruco.DICT_4X4_100,
            "5X5_100": cv2.aruco.DICT_5X5_100,
            "5X5_250": cv2.aruco.DICT_5X5_250,
        }
        # 创建ChArUco标定板
        dictionary = cv2.aruco.getPredefinedDictionary(dict=self.ARUCO_DICT[aruco_type])
        board = cv2.aruco.CharucoBoard_create(
            squaresY=long,
            squaresX=short,
            squareLength=squareLength,
            markerLength=aruco_len,
            dictionary=dictionary,
        )
        corners = board.chessboardCorners
        # ATTN: exchange the XY
        corners = corners[:, [1, 0, 2]]
        self.template = {
            'keypoints3d': corners,
            'keypoints2d': np.zeros_like(corners),
            'pattern': (long-1, short-1),
            'grid_size': squareLength,
            'visted': False
        }
        print(corners.shape)
        self.dictionary = dictionary
        self.board = board
    
    def detect(self, img_color, annots):
        # 查找标志块的左上角点
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img_color, dictionary=self.dictionary, parameters=None
        )
        # 棋盘格黑白块内角点
        if ids is not None:
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=img_color, board=self.board
            )
        else:
            retval = False
        if retval:
            # 绘制棋盘格黑白块内角点
            cv2.aruco.drawDetectedCornersCharuco(
                img_color, charucoCorners, charucoIds, [0, 0, 255]
            )
            if False:
                cv2.aruco.drawDetectedMarkers(
                    image=img_color, corners=corners, ids=ids, borderColor=None
                )

            ids = charucoIds[:, 0]
            pts = charucoCorners[:, 0]
            annots['keypoints2d'][ids, :2] = pts
            annots['keypoints2d'][ids, 2] = 1.
            # if args.show:
            #     img_color = cv2.resize(img_color, None, fx=0.5, fy=0.5)
            #     cv2.imshow('vis', img_color)
            #     cv2.waitKey(0)
            # visname = imgname.replace(images, output)
            # os.makedirs(os.path.dirname(visname), exist_ok=True)
            # cv2.imwrite(visname, img_color)
        else:
            # mywarn('Cannot find in {}'.format(imgname))
            pass
        
    def __call__(self, imgname, images='images', output='output'):
        import os
        from .file_utils import read_json, save_json
        import copy
        img_color = cv2.imread(imgname)
        annotname = imgname.replace('images', 'chessboard').replace('.jpg', '.json')
        if os.path.exists(annotname):
            annots = read_json(annotname)
            if annots['visited']:
                return
        else:
            annots = copy.deepcopy(self.template)
        annots['visited'] = True
        self.detect(img_color, annots)
        annots['keypoints2d'] = annots['keypoints2d'].tolist()
        annots['keypoints3d'] = annots['keypoints3d'].tolist()
        save_json(annotname, annots)