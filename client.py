import aimodel_pb2
import zmq

from common import *


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
            elif count < 200:
                count += 1
                print("no respond {} times, reconnecting".format(count))
                self.socket.close()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect("tcp://localhost:5555")
                self.send_reset()
            else:
                print("send too many times, restart connection.")
                self.socket.close()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect("tcp://localhost:5555")
                return None, False
        return state, True

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
        state[0][0] = msg_env.self.dof.lat
        state[0][1] = msg_env.self.dof.lon
        state[0][2] = 5500  # msg_env.self.dof.height, 因为传输数据和观战不符, 所以先给高度一个固定值
        state[0][3] = 0     # msg_env.self.dof.phi  # roll, 翻滚角
        state[0][4] = 0     # msg_env.self.dof.theta  # pitch, 俯仰角
        state[0][5] = msg_env.self.dof.psi  # yaw, 偏航角
        state[0][6] = msg_env.self.vel.vel_north
        state[0][7] = msg_env.self.vel.vel_east
        state[0][8] = 0     # msg_env.self.vel.vel_down
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
            reward, done = get_reward_done(cur_state, next_state, msg_env.red_crash, msg_env.blue_crash)
            return next_state, reward, done
