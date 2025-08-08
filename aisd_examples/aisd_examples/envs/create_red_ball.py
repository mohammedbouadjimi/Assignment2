import gymnasium as gym
from gymnasium import spaces
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RedBall(Node):
    def __init__(self):
        super().__init__('red_ball_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.bridge = CvBridge()
        self.redball_position = 320  # default center
        self.create3_is_stopped = True  # placeholder for now

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red mask range
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        moments = cv2.moments(mask)

        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            self.redball_position = cx
        else:
            self.redball_position = 320  # fallback center

    def step(self, action):
        twist = Twist()
        twist.angular.z = (action - 320) / 320 * (np.pi / 2)
        self.cmd_pub.publish(twist)
        self.create3_is_stopped = True  # placeholder until you add real /stop_status
       

class CreateRedBallEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Discrete(640)
        self.action_space = spaces.Discrete(640)
        self.step_count = 0

        rclpy.init()
        self.redball = RedBall()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self.redball.redball_position, {}

    def step(self, action):
        self.step_count += 1
        self.redball.step(action)

        # Spin ROS 2 to get camera update
        rclpy.spin_once(self.redball)

        # Optional: Wait for robot to stop if you implement /stop_status subscriber
        while not self.redball.create3_is_stopped:
            rclpy.spin_once(self.redball)

        obs = self.redball.redball_position
        reward = self.reward(obs)
        terminated = self.step_count >= 100
        return obs, reward, terminated, False, {}

    def reward(self, obs):
        # Reward closer to center (320)
        return 1.0 - abs(obs - 320) / 320

    def render(self):
        pass  # no rendering required

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
