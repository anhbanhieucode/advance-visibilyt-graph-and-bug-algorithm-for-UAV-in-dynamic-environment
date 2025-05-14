import time
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from AStarPlanner import AStarPlanner
from Vplanner import DWA

# from planner import AStarPlanner,RRTPlanner
# from localplanner import dwa

plt.rcParams["figure.figsize"] = [10.0, 10.0]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['keymap.save'].remove('s')


def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


class DWAConfig:
    robot_radius = 0.25

    def __init__(self, obs_radius):
        self.obs_radius = obs_radius
        self.dt = 0.1  # [s] Time tick for motion prediction

        self.max_speed = 1.5  # [m/s] 最大线速度
        self.min_speed = -0.5  # [m/s] 最小线速度
        self.max_accel = 0.5 # [m/ss] 加速度
        self.v_reso = self.max_accel*self.dt/10.0  # [m/s] 速度增加的步长

        self.min_yawrate = -100.0 * math.pi / 180.0  # [rad/s] 最小角速度
        self.max_yawrate = 100.0 * math.pi / 180.0  # [rad/s] 最大角速度
        self.max_dyawrate = 500.0 * math.pi / 180.0  # [rad/ss] 角加速度
        self.yawrate_reso = self.max_dyawrate*self.dt/10.0  # [rad/s] 角速度增加的步长

        # 模拟轨迹的持续时间
        self.predict_time = 2  # [s]

        # 三个比例系数
        self.to_goal_cost_gain = 1.0  # 距离目标点的评价函数的权重系数
        self.speed_cost_gain = 0.7  # 速度评价函数的权重系数
        self.obstacle_cost_gain = 1.0  # 距离障碍物距离的评价函数的权重系数

        self.tracking_dist = self.predict_time*self.max_speed
        self.arrive_dist = 0.1


class Playground:
    planning_obs_radius = 0.2

    def __init__(self, planner=None, vplanner=None):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vw = 0.0
        self.x_traj = []
        self.y_traj = []

        self.dwaconfig = DWAConfig(self.planning_obs_radius)
        self.dt = self.dwaconfig.dt

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.mpl_connect('button_press_event', self.on_mousepress)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mousemove)
        self.NEED_EXIT = False

        ############################################

        self.planning_obs = np.empty(shape=(0, 2))

        self.planning_path = np.empty(shape=(0, 2))

        self.planning_target = None

        self.planner = planner
        self.vplanner = vplanner

        self.vplanner_midpos_index = None

        #####################################
        self.temp_obs = [0, 0]

    def run(self):
        while True:
            if self.NEED_EXIT:
                plt.close("all")
                break
            self.vplanner_midpos_index = self.check_path()
            all_traj = []
            all_u = []
            best_traj = None
            if self.vplanner_midpos_index >= 0:
                midpos = self.planning_path[self.vplanner_midpos_index]
                [self.vx,self.vw], best_traj, all_traj, all_u = self.vplanner.plan([self.x,self.y,self.theta,self.vx,self.vw],
                    self.dwaconfig,midpos,self.planning_obs)
            else:
                self.vx, self.vw = 0.0, 0.0
        
            dx, dy, dw = self.vx*self.dt, 0, self.vw*self.dt
            T = transformation_matrix(self.x, self.y, self.theta)
            p = np.matmul(T, np.array([dx, dy, 1]))
            self.x = p[0]
            self.y = p[1]
            self.theta += dw
            self.x_traj.append(self.x)
            self.y_traj.append(self.y)

            plt.cla()
            self.__draw(all_traj, all_u, best_traj=best_traj)

    def check_path(self):
        if self.planning_path is None or self.planning_path.shape[0] == 0:
            return -1
        if self.vplanner_midpos_index is not None and self.vplanner_midpos_index >= 0:
            midindex = self.vplanner_midpos_index
            while True:
                midpos = self.planning_path[midindex]
                dist = np.hypot(self.x-midpos[0], self.y-midpos[1])
                if dist > self.dwaconfig.tracking_dist:
                    break
                if midindex + 1 == self.planning_path.shape[0]:
                    return midindex
                midindex += 1
            return midindex
        else:
            return 0

    def add_obs(self, x, y):
        self.planning_obs = np.append(self.planning_obs, [[x, y]], axis=0)

    def add_obss(self, xs, ys):
        self.planning_obs = np.append(
            self.planning_obs, np.vstack([xs, ys]).T, axis=0)

    def __draw(self, all_traj, all_value, best_traj):
        assert(self.planning_path is None or self.planning_path.shape[1] == 2,
               "the shape of planning path should be '[x,2]', please check your algorithm.")
        assert(self.planning_obs is None or self.planning_obs.shape[1] == 2,
               "the shape of self.planning_obs(obstacles) should be '[x,2]', please check your algorithm.")

        p1_i = np.array([0.5, 0, 1]).T
        p2_i = np.array([-0.5, 0.25, 1]).T
        p3_i = np.array([-0.5, -0.25, 1]).T

        T = transformation_matrix(self.x, self.y, self.theta)
        p1 = np.matmul(T, p1_i)
        p2 = np.matmul(T, p2_i)
        p3 = np.matmul(T, p3_i)

        plt.plot([p1[0], p2[0], p3[0], p1[0]], [
                 p1[1], p2[1], p3[1], p1[1]], 'k-')

        if self.planning_target is not None:
            self.ax.plot(
                self.planning_target[0], self.planning_target[1], "r*", markersize=20)
        
        if len(all_traj) > 0:
            all_value = np.array(all_value,dtype=float)
            all_value = (all_value-all_value.min())/(all_value.max()-all_value.min())
            for i,traj in enumerate(all_traj):
                color = plt.cm.jet(all_value[i])
                self.ax.plot(traj[:,0],traj[:,1],".",color=color,markersize=1)
                self.ax.plot(traj[-1,0],traj[-1,1],"+",color=color,markersize=3)

        if best_traj is not None:
            self.ax.plot(best_traj[:,0],best_traj[:,1],color="green",linewidth=3) 

        if self.planning_path is not None:
            self.ax.plot(self.planning_path[:, 0],
                         self.planning_path[:, 1], 'b--')
            if self.vplanner_midpos_index is not None and self.vplanner_midpos_index >= 0:
                midpos = self.planning_path[self.vplanner_midpos_index]
                self.ax.plot(midpos[0], midpos[1], "g+", markersize=20)
        if len(self.x_traj) > 0:
            plt.plot(self.x_traj, self.y_traj, 'g-')
        for obs in self.planning_obs:
            self.ax.add_artist(plt.Circle(
                (obs[0], obs[1]), self.planning_obs_radius, color = 'purple' , fill=True))

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        plt.pause(self.dt)

    def on_mousepress(self, event):
        if not event.dblclick:
            if event.button == 1:
                self.x, self.y = event.xdata, event.ydata
            if event.button == 3:
                self.planning_target = np.array([event.xdata, event.ydata])
            if event.button == 2:
                self.add_obs(event.xdata, event.ydata)
                self.temp_obs = [event.xdata, event.ydata]

    def on_mousemove(self, event):
        if hasattr(event, "button") and event.button == 2:
            dx = event.xdata-self.temp_obs[0]
            dy = event.ydata-self.temp_obs[1]
            if np.hypot(dx, dy) > self.planning_obs_radius*0.8:
                self.temp_obs = [event.xdata, event.ydata]
                self.add_obs(*self.temp_obs)

    def on_press(self, event):
        if(event.key == 'escape'):
            self.set_exit()
        if(event.key == ' '):
            self.planning_path = None
            self.x_traj, self.y_traj = [], []
            self.vplanner_midpos_index = None
            if self.planning_target is not None and self.planner is not None:
                print("do planning...")
                px, py = planner.planning(self.planning_obs[:, 0], self.planning_obs[:, 1], Playground.planning_obs_radius +
                                          DWAConfig.robot_radius, self.x, self.y, self.planning_target[0], self.planning_target[1], -10, -10, 10, 10)
                self.planning_path = np.vstack([px, py]).T
                print("pathLength : ", self.planning_path.shape[0])

    def set_exit(self):
        self.NEED_EXIT = True


if __name__ == "__main__":
    planner = AStarPlanner(0.2)
    vplanner = DWA()
    pg = Playground(planner, vplanner)
    
    # Set start and goal points
    pg.x, pg.y = 1, 1  # Start point
    pg.planning_target = np.array([9, 9])  # Goal point
    # Add circular obstacles
  
    preset_purple_circles_10 = [
    [1.2, 7.8], [8.3, 2.4], [3.5, 5.7], [9.1, 3.9], [2.8, 9.2],
    [5.4, 3.8], [7.9, 8.4], [6.1, 2.5], [4.7, 9.6], [1.0, 6.3]
    
    ]

    preset_purple_circles_50 = [
    [2.5, 9.7], [7.8, 3.2], [4.1, 5.6], [9.3, 2.9], [0.7, 6.1],
    [3.6, 8.5], [5.9, 4.7], [1.4, 2.0], [8.2, 6.4], [9.0, 1.1],
    [6.8, 8.2], [7.1, 3.5], [4.9, 9.4], [6.3, 1.7], [2.1, 7.0],
    [0.3, 8.3], [8.0, 4.9], [1.9, 2.2], [9.6, 5.1], [3.0, 6.6],
    [4.3, 1.8], [7.6, 9.3], [6.7, 2.6], [0.9, 7.4], [8.7, 3.0],
    [3.8, 9.9], [7.5, 5.3], [5.5, 6.8], [1.2, 4.0], [9.4, 2.7],
    [0.6, 3.9], [6.9, 0.8], [2.4, 6.9], [8.9, 5.8], [3.7, 1.2],
    [7.2, 4.2], [1.3, 5.0], [4.0, 2.3], [9.8, 6.0], [5.2, 3.4],
    [8.4, 9.5], [0.2, 4.4], [3.9, 7.3], [6.2, 1.9], [9.7, 8.1],
    [1.8, 2.8], [7.4, 6.5], [4.6, 8.1], [2.3, 9.1], [5.8, 4.5],
    [6.0, 7.9], [9.2, 3.3], [3.3, 0.5], [7.3, 1.5], [2.7, 7.2]
    ]

    preset_purple_circles_100 = [
    [1.4, 2.7], [4.1, 9.2], [6.5, 1.9], [2.6, 4.4], [3.9, 8.7],
    [0.8, 6.0], [7.2, 5.3], [6.3, 9.6], [4.7, 1.5], [1.0, 2.3],
    [8.1, 7.5], [3.0, 9.3], [7.4, 4.0], [6.9, 3.1], [5.2, 8.9],
    [8.3, 1.0], [9.8, 2.9], [1.2, 6.2], [9.6, 5.1], [0.7, 8.4],
    [2.0, 1.3], [6.4, 4.9], [5.0, 7.6], [9.3, 3.8], [7.9, 0.2],
    [8.5, 6.8], [3.3, 0.9], [6.6, 2.5], [9.7, 9.9], [1.9, 8.2],
    [7.3, 4.6], [2.2, 3.8], [5.4, 2.0], [0.5, 9.0], [8.0, 5.6],
    [9.4, 7.1], [6.8, 1.8], [4.0, 7.8], [3.7, 4.2], [2.4, 6.6],
    [8.9, 8.3], [7.7, 0.3], [5.5, 9.1], [6.2, 5.7], [3.8, 0.7],
    [0.3, 4.8], [9.2, 2.1], [4.5, 7.2], [7.0, 3.6], [8.6, 1.2],
    [2.5, 9.8], [1.3, 5.5], [3.6, 2.9], [5.1, 0.4], [6.0, 6.7],
    [2.9, 3.5], [9.5, 8.0], [7.1, 7.7], [5.9, 4.3], [8.4, 3.2],
    [4.4, 1.7], [0.9, 6.3], [7.8, 8.8], [2.1, 4.5], [1.6, 2.8],
    [5.3, 0.8], [4.2, 5.0], [9.0, 1.4], [2.7, 0.6], [6.7, 9.0],
    
    ]



    circular_obstacles = [
        (3, 3), (5, 1), (4, 4), (2, 2)
    ]
    for obs_x, obs_y in preset_purple_circles_10:
        pg.add_obs(obs_x, obs_y)  # Circular obstacles
    '''
    #### Add rectangular obstacles defined by (x, y, width, height)
    rectangles = [
        (-6, -1, 2, 2),  # Rectangle 1 at (-6, -1) with width=2, height=2
        (2, -6, 2, 2)    # Rectangle 2 at (2, -6) with width=2, height=2
    ]
    '''
    '''
    for (rect_x, rect_y, width, height) in rectangles:
        # Calculate the four corners based on (x, y, width, height)
        rect_points = [
            (rect_x, rect_y),  # Bottom-left corner
            (rect_x + width, rect_y),  # Bottom-right corner
            (rect_x + width, rect_y + height),  # Top-right corner
            (rect_x, rect_y + height)  # Top-left corner
        ]
        xs, ys = zip(*rect_points)
        pg.add_obss(xs, ys)  # Adding rectangle as a set of boundary points
    '''
    # Run the simulation
    pg.run()
