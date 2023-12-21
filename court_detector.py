import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
from court_reference import CourtReference
import scipy.signal as sp


class CourtDetector:
    """
    Detecting and tracking court in frame
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line = None
        self.top_inner_line = None
        self.bottom_inner_line = None
        self.success_flag = False
        self.success_accuracy = 80
        self.success_score = 1000
        self.best_conf = None
        self.frame_points = None
        self.dist = 2
        self.gray_threshold = 170
        self.ori_lines = np.array(self.court_reference.get_table_lines(), dtype=np.float32).reshape((-1, 1, 2))

    def detect_table(self, frame, verbose=0):
        """
        Detecting the court in the frame
        """
        cv2.imwrite('frame.jpg',frame)
        self.verbose = verbose
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        #Get gray image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray.jpg',gray)
        # Get binary image from the frame
        self.bin = cv2.threshold(gray, self.gray_threshold, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('bin.jpg',self.bin)
        # Filter pixel
        filtered = self._filter_pixels(self.bin)
        cv2.imwrite('filtered.jpg',filtered)
        # Detect horizontal_lines & vertical_lines
        horizontal_lines, vertical_lines = self._detect_lines(filtered)
        if horizontal_lines is None or len(horizontal_lines)<2:
            self.court_warp_matrix.append(None)
            self.game_warp_matrix.append(None)
            return None
        if vertical_lines is None or len(vertical_lines)<2:
            self.court_warp_matrix.append(None)
            self.game_warp_matrix.append(None)
            return None
        # Find transformation matrix from reference_court to frame_court
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal_lines,vertical_lines)
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        self.tra_lines = cv2.perspectiveTransform(self.ori_lines, self.court_warp_matrix[-1]).reshape(-1)
        return self.tra_lines.astype(int)


    def _filter_pixels(self, gray):  #留下白色邊緣
        """
        Filter pixels by using the court line structure
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue
                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        """
        Finds all line in frame using Hough transform
        """
        # Detect all lines
        lines = cv2.HoughLinesP(gray, 1.0, np.pi / 180, 75, minLineLength=40, maxLineGap=70)

        if lines is None:
            return None,None
        #print('lines: ',lines)
        #print('lines.shape: ',lines.shape)
        lines = np.reshape(lines,(-1,4))
        #print('squeze lines: ',lines)
        #print('squeze lines.shape: ',lines.shape)
        HL = display_lines_on_frame(self.frame.copy(), [], lines)
        cv2.imwrite('HoughLinesP.jpg',HL)

        horizontal, vertical = self._classify_lines(lines)
        if horizontal==[] or vertical==[]:
            return None,None
        CL = display_lines_on_frame(self.frame.copy(), horizontal, vertical)
        cv2.imwrite('classify_lines.jpg',CL)

        horizontal, vertical = self._merge_lines(horizontal, vertical)
        if horizontal==[] or vertical==[]:
            return None,None
        ML = display_lines_on_frame(self.frame.copy(), horizontal, vertical)
        cv2.imwrite('merge_lines.jpg',ML)
        return horizontal, vertical

    def _classify_lines(self, lines):
        """
        Classify line to vertical and horizontal lines
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # Filter horizontal lines using vertical lines lowest and highest point
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)

        return clean_horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        """
        Merge lines that belongs to the same frame`s lines
        """
        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        Finds transformation from reference court to frame`s court using 4 pairs of matching points
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        # Loop over every pair of horizontal lines and every pair of vertical lines
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            h1, h2 = horizontal_pair
            for vertical_pair in list(combinations(vertical_lines, 2)):
                v1, v2 = vertical_pair
                # Finding intersection points of all lines
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)

                for i, configuration in self.court_reference.court_conf.items():

                    matrix = cv2.getPerspectiveTransform(np.float32(configuration), np.float32(intersections))
                    inv_matrix = cv2.invert(matrix)[1]
                    # Get confidence score
                    court = cv2.warpPerspective(self.court_reference.court, matrix, (self.v_width, self.v_height))
                    bin_img = self.bin.copy()
                    confi_score = self._get_confi_score(court, bin_img)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i

        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, court, bin_image):
        confi_score = court * bin_image
        #wrong = court - correct
        #c_p = np.sum(correct)
        #w_p = np.sum(wrong)
        #confi_score = c_p - 0.3 * w_p
        return confi_score

    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def get_warped_court(self):
        """
        Returns warped court using the reference court and the transformation of the court
        """
        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def track_table(self, frame):
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_points is None:
            conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points,self.court_warp_matrix[-1]).squeeze().round()
        # Lines of configuration on frames
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]

        ok = 0
        for line in lines:
            # Get 100 samples of each line in the frame
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line

            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # if one of the ends of the line is out of the frame get only the points inside the frame
            points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[1:-1]

            new_points = []
            # Find max intensity pixel near each point
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))

                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]

                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > self.gray_threshold:
                    new_p = (x + top_x, y + top_y)
                    #cv2.circle(copy,new_p,4,(0,255,0),-1)
                    new_points.append(new_p)
            #
            if len(new_points)>40:
                ok += 1
        #
        if ok < 3:
            return self.detect(frame)
        self.court_warp_matrix.append(self.court_warp_matrix[-1])
        self.game_warp_matrix.append(self.game_warp_matrix[-1])
        return self.tra_lines.astype(int)

    def original_track_court(self, frame):
        """
        Track court location after detection
        """
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_points is None:
            conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points,self.court_warp_matrix[-1]).squeeze().round()
        # Lines of configuration on frames
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]
        new_lines = []
        for line in lines:
            # Get 100 samples of each line in the frame
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line

            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # if one of the ends of the line is out of the frame get only the points inside the frame
            points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[1:-1]

            new_points = []
            # Find max intensity pixel near each point
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))

                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]

                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > self.gray_threshold:
                    new_p = (x + top_x, y + top_y)
                    #cv2.circle(copy,new_p,4,(0,255,0),-1)
                    new_points.append(new_p)

            print('len(new_points): ',len(new_points))

            #若是能track出的new_points少於50點的話,就運行detect()
            if len(new_points)<50:
                return self.detect(frame)

            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
            # find line fitting the new points
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),(int(x + vx * self.v_width), int(y + vy * self.v_width))))

        # Find transformation from new lines
        i1 = line_intersection(new_lines[0], new_lines[2])
        i2 = line_intersection(new_lines[0], new_lines[3])
        i3 = line_intersection(new_lines[1], new_lines[2])
        i4 = line_intersection(new_lines[1], new_lines[3])

        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        matrix, _ = cv2.findHomography(np.float32(self.court_reference.court_conf[1]),intersections, method=0)
        inv_matrix = cv2.invert(matrix)[1]
        self.court_warp_matrix.append(matrix)
        self.game_warp_matrix.append(inv_matrix)
        self.frame_points = intersections
        self.pts = np.array(self.court_reference.get_table_lines(), dtype=np.float32).reshape((-1, 1, 2))
        self.new_court = cv2.perspectiveTransform(self.pts, self.court_warp_matrix[-1]).reshape(-1)
        return self.new_court

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates

def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34

def lines_overlay(l1,l2):
    if l1[0]==l2[0] and l1[2]==l2[2]:
        return True
    if l1[1]==l2[1] and l1[3]==l2[3]:
        return True

def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
     Display lines on frame for horizontal and vertical lines
    """
    for line in horizontal:
       x1, y1, x2, y2 = line
       cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
       cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
       cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
       x1, y1, x2, y2 = line
       cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
       cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
       cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)
    return frame
