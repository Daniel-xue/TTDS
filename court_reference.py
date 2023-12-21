import cv2
import numpy as np
import matplotlib.pyplot as plt

class CourtReference:
    """
    Court reference model
    """
    def __init__(self):
        self.baseline_top = ((0,0), (152, 0))#上
        self.baseline_bottom = ((0, 274), (152, 274))#底
        self.left_line = ((0,0),(0,274))#左
        self.right_line = ((152,0),(152,274))#右
        self.middle_line = ((76, 0), (76, 274))#中

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.middle_line[0], self.right_line[0],self.middle_line[1],self.right_line[1]],
                           3: [self.left_line[0],self.middle_line[0],self.left_line[1],self.middle_line[1]]
                           }
        self.court = cv2.imread('table_border.png')

        self.line_width = 2
        self.court_width = 823
        self.court_height = 458
        self.top_bottom_border = 548.5
        self.right_left_border = 311
        self.court_total_width = 152+1
        self.court_total_height = 274+1

    def build_table_reference(self):
        """
        Create table reference image using the lines positions
        """
        table = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)
        cv2.line(table, *self.baseline_top, 1, self.line_width)
        cv2.line(table, *self.baseline_bottom, 1, self.line_width)
        cv2.line(table, *self.left_line, 1, self.line_width)
        cv2.line(table, *self.right_line, 1, self.line_width)
        cv2.line(table, *self.middle_line, 1, self.line_width)

        table = cv2.dilate(table, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('table_border.png', table, cmap='gray')
        self.court = table
        return table

    def get_table_lines(self):
        """
        Returns all lines of the court
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.left_court_line, *self.right_court_line]

        return lines

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court
            mask[:self.net[0][1] - 1000, :] = 0
        elif mask_type == 2:  # Top half court
            mask[self.net[0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            mask[:self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1]:, :] = 0
            mask[:, :self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0]:] = 0
        return mask


if __name__ == '__main__':
    c = CourtReference()
    c.build_table_reference()
