from torch.nn.modules.module import register_module_backward_hook
import aimodel_pb2
import numpy as np
import math
import zmq

from config import *


class MyClient:

    # 建立TCP连接
    def __init__(self):
        self.context = zmq.Context()
        print("Connecting to hello world server...")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def send_reset(self):
        # print(f"Sending request 'reset' ..." )

        msg_action = aimodel_pb2.Action()
        msg_action.isReset = True
        send_msg = msg_action.SerializeToString()
        # print(f"Sending request reset:{msg_action.isReset} ...")

        self.socket.send(send_msg)

    # 发送动作信息
    def send_action(self, action):
        action = action.astype(np.float64)

        msg_action = aimodel_pb2.Action()
        msg_action.isReset = False
        msg_action.point.lat = action[0]
        msg_action.point.lon = action[1]
        msg_action.point.h = action[2]
        msg_action.point.vel = 250
        msg_action.point.ref_phi = 0

        send_msg = msg_action.SerializeToString()
        self.socket.send(send_msg)

    def poll_reset(self):
        count = 0
        while True:
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            sockets = dict(poller.poll(2500))
            if self.socket in sockets:
                state, _, _ = self.recv_step(None)
                break
            else:
                count += 1
                print("no respond {} times, reconnecting".format(count))
                self.socket.close()
                print("reconnecting...")
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect("tcp://localhost:5555")
                self.send_reset()
        return state

    def recv_step(self, cur_state):
        msg_env = aimodel_pb2.Env()
        rev_msg = self.socket.recv()
        msg_env.ParseFromString(rev_msg)
        '''
        0   自己纬度 
        1   自己经度    
        2   自己高度
        3   自己翻滚角
        4   自己俯仰角
        5   自己偏航角
        6   自己南北速度
        7   自己东西速度
        8   自己上下速度
        ADD
        9   自己的武器数量 (bool)
        ### 自己的导弹信息无法获取
        10  雷达信息 (常开)  (bool)
        11  目标点纬度
        12  目标点经度
        13  目标点高度
        14  敌人纬度
        15  敌人经度
        16  敌人高度
        17  敌人翻滚角 
        18  敌人俯仰角
        19  敌人偏航角
        20  敌人南北速度
        21  敌人东西速度
        22  敌人上下速度
        23  导弹距离
        24  导弹方向
        '''

        state = np.zeros((1, state_dim))
        print("lat:", msg_env.self.dof.lat)
        state[0][0] = msg_env.self.dof.lat
        state[0][1] = msg_env.self.dof.lon
        state[0][2] = msg_env.self.dof.height
        state[0][3] = msg_env.self.dof.phi  # roll, 翻滚角
        state[0][4] = msg_env.self.dof.theta  # pitch, 俯仰角
        state[0][5] = msg_env.self.dof.psi  # yaw, 偏航角
        state[0][6] = msg_env.self.vel.vel_north
        state[0][7] = msg_env.self.vel.vel_east
        state[0][8] = msg_env.self.vel.vel_down
        state[0][9] = 1  # msg_env.num_wpn  # TODO 想一想这里是设计成bool还是int, 暂时先空着因为没有attack
        state[0][10] = msg_env.radar_on
        state[0][11] = msg_env.goal.lat
        state[0][12] = msg_env.goal.lon
        state[0][13] = msg_env.goal.height
        state[0][14] = msg_env.enemy.dof.lat if msg_env.detect_enemy else -1
        state[0][15] = msg_env.enemy.dof.lon if msg_env.detect_enemy else -1
        state[0][16] = msg_env.enemy.dof.height if msg_env.detect_enemy else -1
        state[0][17] = msg_env.enemy.dof.phi if msg_env.detect_enemy else -1
        state[0][18] = msg_env.enemy.dof.theta if msg_env.detect_enemy else -1
        state[0][19] = msg_env.enemy.dof.psi if msg_env.detect_enemy else -1
        state[0][20] = msg_env.enemy.vel.vel_north if msg_env.detect_enemy else -1
        state[0][21] = msg_env.enemy.vel.vel_east if msg_env.detect_enemy else -1
        state[0][22] = msg_env.enemy.vel.vel_down if msg_env.detect_enemy else -1
        state[0][23] = msg_env.missle.dist if msg_env.detect_missle else -1
        state[0][24] = msg_env.missle.dir if msg_env.detect_missle else -1

        next_state = state[0]
        if cur_state is None:
            return next_state, 0, 0
        else:
            reward, done = self.get_reward_done(cur_state, next_state, msg_env.red_crash, msg_env.blue_crash)
            return next_state, reward, done

    def get_reward_done(self, cur_state, next_state, red_crash, blue_crash):
        reward, done = 0, False
        # done的情况: 红方被击毁, 蓝方被击毁, 红方到达目标点, 红方坠机, 到达最长时间

        # 根据目标点距离给予连续性小奖励
        goal_lat, goal_lon, goal_height = cur_state[11], cur_state[12], cur_state[13]
        cur_dist = math.sqrt(pow(cur_state[0] - goal_lat, 2) + pow(cur_state[1] - goal_lon, 2))  # 乘以一个数量级, 防止dist过小
        next_dist = math.sqrt(pow(next_state[0] - goal_lat, 2) + pow(next_state[1] - goal_lon, 2))  # 乘以一个数量级, 防止dist过小
        print("haha")
        print(cur_state[0], cur_state[1], next_state[0], next_state[1])
        print(goal_lat, goal_lon)
        print(cur_dist, next_dist, cur_dist - next_dist)
        print("xixi")
        reward += cur_dist - next_dist

        # TODO 根据导弹的范围来设计奖励, 但是导弹并不是全局信息, 所以需要考虑疏忽的情况

        # 到达目标点给予一次性大奖励
        if cur_dist < 0.05:  # TODO 根据距离来更改阈值
            print("arrive goal!")
            reward += 5
            done = True

        # 红方被击中, 给予一次性大惩罚
        if red_crash:
            print("red crash!")
            reward -= 5
            done = True

        # 蓝方被击中, 给予一次性大奖励
        if blue_crash:
            print("blue crash!")
            reward += 5

        # 坠机, 给予一次性大惩罚
        delta_theta = abs(next_state[4] - cur_state[4])
        if delta_theta > 180:
            delta_theta = 360 - delta_theta
        delta_psi = abs(next_state[5] - cur_state[5])
        if delta_psi > 180:
            delta_psi = 360 - delta_psi
        if delta_theta > 15 or delta_psi > 15:
            print("crash down! ", delta_theta, delta_psi, next_state[4], cur_state[4], next_state[5], cur_state[5])
            reward -= 5
            done = True
        return reward, done