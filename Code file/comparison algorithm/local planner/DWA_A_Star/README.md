# Robot
- 实现Astar算法和DWA算法的结合
1. main.py: 文件可以通过Astar算法实现两点间的路径规划
2. dwa.py: 文件在main.py文件的基础上增加了dwa动态窗口算法，可以实现小车在运行过程中动态避障功能
3. Vplanner.py: dwa算法实现
4. AStarPlanner.py: astar算法实现

- 关键控制指令：
1. 按下鼠标左键放置起始点
2. 按下鼠标右键放置终点
3. 按下鼠标中键放置障碍物
4. 按下空格键开始规划路径


## Astar算法实现的示意图

<img src="https://user-images.githubusercontent.com/85838942/163850711-3e2e84cd-3db7-45d9-9d0d-265adc01f635.jpg" width=50% height=50%>



## dwa算法实现的示意图
<img src="https://user-images.githubusercontent.com/85838942/163850489-1254575b-d0e3-4c39-a9c6-983b36ad43d3.jpg" width=70% height=70%>

## dwa算法带雷达显示

<img src="https://user-images.githubusercontent.com/85838942/163851756-2192d93f-a3ff-49cb-9bc2-154c76d32d9c.jpg" width=50% height=50%>


由于dwa算法的缺陷，容易陷入局部最优解，改进的方法是在上述情况下重新规划路径，实现动态dwa

## Gazebo仿真环境

<img src="https://user-images.githubusercontent.com/85838942/164475980-2c5ee1d4-1a9c-4fde-b382-41fcfbcb8b1d.png" width=50% height=50%>

<img src="https://user-images.githubusercontent.com/85838942/164475994-257a65a5-8b61-4313-9155-887bef3fd7a9.png" width=50% height=50%>



