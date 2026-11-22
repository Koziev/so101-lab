"""
ChArUco board shared by the wrist-camera tools: a checkerboard whose white squares carry ArUco markers, so
corners are identified even when only part of the board is visible or a jaw covers some of it.

Default board (print charuco_8x6_15mm_A4.pdf at 100 %): 8 x 6 squares of 15 mm, 11 mm markers, DICT_4X4_50
-> 7 x 5 = 35 inner corners. Requires OpenCV >= 4.7 (cv2.aruco.CharucoDetector).
"""
import cv2
import numpy as np

SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM, DICT = 8, 6, 15.0, 11.0, cv2.aruco.DICT_4X4_50


def make_board(squares_x=SQUARES_X, squares_y=SQUARES_Y, square_mm=SQUARE_MM, marker_mm=MARKER_MM, dict_id=DICT):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0, d)


def detect(board, gray):
    """-> (object points Nx3 in metres, image points Nx1x2) of the detected ChArUco corners, or (None, None)."""
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None, None
    obj, img = board.matchImagePoints(corners, ids)
    return np.asarray(obj, np.float32).reshape(-1, 3), np.asarray(img, np.float32).reshape(-1, 1, 2)


if __name__ == "__main__":                      # write the printable PDF
    from PIL import Image, ImageDraw, ImageFont
    dpi = 300
    px_mm = dpi / 25.4
    b = make_board()
    pat = b.generateImage((int(SQUARES_X * SQUARE_MM * px_mm), int(SQUARES_Y * SQUARE_MM * px_mm)), marginSize=0)
    page = Image.new("L", (int(210 * px_mm), int(297 * px_mm)), 255)
    x0, y0 = int((210 - SQUARES_X * SQUARE_MM) / 2 * px_mm), int(40 * px_mm)
    page.paste(Image.fromarray(pat), (x0, y0))
    d = ImageDraw.Draw(page)
    yb = y0 + pat.shape[0] + int(25 * px_mm)
    xb = int(55 * px_mm)
    d.rectangle([xb, yb, xb + int(100 * px_mm), yb + int(2 * px_mm)], fill=0)
    for k in range(11):
        d.rectangle([xb + int(k * 10 * px_mm), yb - int(3 * px_mm), xb + int(k * 10 * px_mm) + 3, yb], fill=0)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        f = ImageFont.load_default()
    d.text((xb, yb + int(6 * px_mm)), "100 mm - check with a ruler; squares must be 15.0 mm", font=f, fill=0)
    d.text((x0, int(20 * px_mm)), f"ChArUco {SQUARES_X}x{SQUARES_Y} squares, {SQUARE_MM:g} mm, markers {MARKER_MM:g} mm, "
           "DICT_4X4_50. Print at 100 %.", font=f, fill=0)
    page.save("charuco_8x6_15mm_A4.pdf", resolution=dpi)
    print("written charuco_8x6_15mm_A4.pdf")


# ------------------------------------------------------------------------------------------------ board sets
# Several independent boards with disjoint marker ids (each sheet is rigid on its own; their relative placement
# does not matter: the calibration estimates each board's pose).
BOARD_SETS = {
    "2xA4": {"squares": (7, 10), "square_mm": 25.0, "marker_mm": 18.0, "dict": cv2.aruco.DICT_4X4_100,
             "boards": ["A", "B"], "pdf": "charuco_2xA4_7x10_25mm.pdf"},
    "wrist_small": {"squares": (SQUARES_X, SQUARES_Y), "square_mm": SQUARE_MM, "marker_mm": MARKER_MM,
                    "dict": DICT, "boards": ["single"]},
}


def make_board_set(name="2xA4"):
    """[(label, board)] with disjoint marker ids (board k uses ids k*n .. (k+1)*n-1)."""
    cfg = BOARD_SETS[name]
    sx, sy = cfg["squares"]
    d = cv2.aruco.getPredefinedDictionary(cfg["dict"])
    n = (sx * sy) // 2
    out = []
    for k, label in enumerate(cfg["boards"]):
        ids = np.arange(k * n, (k + 1) * n, dtype=np.int32)
        out.append((label, cv2.aruco.CharucoBoard((sx, sy), cfg["square_mm"] / 1000, cfg["marker_mm"] / 1000, d, ids)))
    return out


def detect_set(boards, gray, min_corners=8, use_markers=True):
    """[(board index, obj Nx3, img Nx1x2)] for every board of the set yielding >= min_corners points.
    Points: the ChArUco corners (interpolated between markers) and, with use_markers, the four corners of every
    decoded ArUco marker (known board positions, refined to sub-pixel). In close-up views of a board with big
    squares only 2-3 markers are visible and few or no ChArUco corners can be interpolated; the marker corners
    keep such views usable (2 markers = 8 points)."""
    out = []
    d = boards[0][1].getDictionary()
    det = cv2.aruco.ArucoDetector(d)
    m_corners, m_ids, _ = det.detectMarkers(gray) if use_markers else (None, None, None)
    for k, (_, b) in enumerate(boards):
        objs, imgs = [], []
        obj, img = detect(b, gray)
        if obj is not None:
            objs.append(obj.reshape(-1, 3))
            imgs.append(img.reshape(-1, 2))
        if use_markers and m_ids is not None:
            ids = list(np.asarray(b.getIds()).ravel())
            obj_pts = b.getObjPoints()
            for c, i in zip(m_corners, np.asarray(m_ids).ravel()):
                if int(i) in ids:
                    pts = np.asarray(c, np.float32).reshape(4, 1, 2)
                    cv2.cornerSubPix(gray, pts, (3, 3), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
                    objs.append(np.asarray(obj_pts[ids.index(int(i))], np.float32).reshape(4, 3))
                    imgs.append(pts.reshape(4, 2))
        if objs:
            o, im = np.vstack(objs), np.vstack(imgs)
            if len(o) >= min_corners:
                out.append((k, o, im.reshape(-1, 1, 2)))
    return out
