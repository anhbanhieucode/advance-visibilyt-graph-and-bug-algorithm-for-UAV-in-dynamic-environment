"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 100) #50
        self.y_range = (0, 100)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            #[0, 0, 1, 100],
            #[0, 100, 100, 1],
            #[1, 0, 100, 1],
            #[100, 1, 1, 100]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        '''
        obs_rectangle = [
            #[14, 12, 8, 2],
            #[18, 22, 8, 3],
            #[26, 7, 2, 12],
            #[32, 14, 10, 2]

        #[2, 2, 4, 3],       # Medium rectangle in bottom-left
        #[10, 2, 6, 2],      # Long horizontal rectangle in lower-middle
        #[20, 2, 2, 4],      # Small vertical rectangle in bottom-right
        #[2, 10, 3, 6],      # Long vertical rectangle in middle-left
        #[10, 10, 4, 4],     # Square in the center
        #[22, 10, 5, 2],     # Thin horizontal rectangle in middle-right
        #[2, 22, 3, 2],      # Small rectangle in top-left
        #[10, 22, 8, 1],     # Long horizontal rectangle in top-middle
        #[22, 22, 4, 3],     # Medium rectangle in top-right
        #[16,8, 2 , 12]
            [34, 1, 5, 12], [46, 1, 9, 7], [58, 1, 8, 8],
    [70, 1, 7, 6], [82, 1, 8, 10], [94, 1, 6, 5],
    [2, 15, 7, 8], [10, 15, 8, 5], [22, 15, 6, 8],
    [34, 15, 12, 3], [50, 15, 5, 7], [58, 15, 7, 6],
    [70, 15, 9, 4], [82, 15, 8, 5], [94, 15, 6, 8],
    [2, 30, 5, 6], [10, 30, 10, 3], [25, 30, 8, 6],
    [34, 30, 4, 10], [46, 30, 5, 7], [58, 30, 6, 5],
    [70, 30, 9, 4], [82, 30, 6, 5], [94, 30, 5, 10],
    [2, 45, 10, 9], [14, 45, 5, 4], [25, 45, 6, 8],
    [34, 45, 8, 5], [46, 45, 7, 6], [58, 45, 5, 7],
    [70, 45, 6, 4], [82, 45, 9, 5], [94, 45, 6, 5],
    [2, 60, 5, 6], [12, 60, 5, 12], [26, 60, 7, 5],
    [34, 60, 4, 9], [46, 60, 11, 3], [58, 60, 6, 8],
    [70, 60, 9, 4], [82, 60, 5, 5], [94, 60, 7, 8],
    [2, 75, 8, 5], [12, 75, 6, 10], [22, 75, 5, 7],
    [34, 75, 7, 4], [46, 75, 10, 5], [58, 75, 5, 8],
    [70, 75, 6, 7], [82, 75, 8, 5], [94, 75, 10, 4],
    [2, 90, 6, 5], [10, 90, 8, 10], [20, 90, 4, 8],
    [30, 90, 12, 3], [45, 90, 5, 6], [58, 90, 7, 4],
    [70, 90, 8, 5], [82, 90, 10, 4], [94, 90, 6, 5]
        ]
    '''
    
        obs_rectangle =  [
      [4,30,3,30],
   [14,27,3,35],
   [24,28,3,35],
   [34,20,3,38],
   [44,30,3,36],
   [54,26,3,30],
   [64,30,3,30],
   [74,40,3,40],
   [84,30,3,30],
   [94,30,3,35],

   [10,10,10,5],
   [60,5,10,7],
   [20,80,10,8],
   [50,85,15,7],
   [80,14,15,8],
   [4,70,10,5]
    ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            #[7, 12, 3],
            #[46, 20, 2],
            #[15, 5, 2],
            #[37, 7, 3],
            #[37, 23, 3]
        ]

        return obs_cir
