from torch.nn.modules.module import register_module_backward_hook
import aimodel_pb2
import zmq
import numpy as np

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
                state, _, _ = self.recv_state()
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

    def recv_step(self):
        msg_env = aimodel_pb2.Env()
        rev_msg = self.socket.recv()
        msg_env.ParseFromString(rev_msg)

        state = np.zeros((1, state_dim))
        state[0][0] = msg_env.self.dof.lat
        state[0][1] = msg_env.self.dof.lon
        state[0][2] = msg_env.self.dof.height
        state[0][3] = msg_env.self.dof.phi  # roll, 翻滚角
        state[0][4] = msg_env.self.dof.theta  # pitch, 俯仰角
        state[0][5] = msg_env.self.dof.psi  # yaw, 偏航角
        state[0][6] = msg_env.self.vel.vel_north
        state[0][7] = msg_env.self.vel.vel_east
        state[0][8] = msg_env.self.vel.vel_down

        reward = self.get_reward_done(msg_env.reward, )
        # TODO dist需要打印一下, msg_env的enemy需要打印一下
        done = msg_env.done

        return state[0], reward, done

    def get_reward_done(dist, delta_theta, delta_psi, env_done):
        reward, done = 0, env_done
        reward += 100 / (1 + dist)
        if delta_theta > 15 or delta_psi > 15:
            print("crash down! ", delta_theta, delta_psi)
            reward -= 5
            done = True
        return reward, done