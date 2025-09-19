import airsim
import gym
from gym import spaces
import numpy as np
import random
import time
import math
from PIL import Image
import cv2
from airsim import Vector3r, Pose, Quaternionr

class DroneFollowerEnv(gym.Env):
    def __init__(self, ip_address, step_length=1.0, image_shape=(84, 84, 1), max_steps=500, leader_behavior="random"):
        super().__init__()

        self.ip_address = ip_address
        self.image_shape = image_shape
        self.step_length = step_length
        self.max_steps = max_steps
        self.leader_behavior = leader_behavior
        self.step_count = 0
        self.obstacle_drones = ["Obstacle0", "Obstacle1", "Obstacle2", "Obstacle3", "Obstacle4"]



        # ✅ 변경: 통합된 이미지 입력
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=image_shape, dtype=np.float32),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.optimal_distance_range = (0, 10)
        self.max_forward_speed = 8.0

        self.drone1 = airsim.MultirotorClient(ip=self.ip_address)
        self.drone2 = airsim.MultirotorClient(ip=self.ip_address)

        self.image_requests = [
            airsim.ImageRequest("depth_front", airsim.ImageType.DepthPerspective, True, False),
            airsim.ImageRequest("depth_back", airsim.ImageType.DepthPerspective, True, False)
        ]

        self._first_setup = True
        self._setup_flight()

        self.base_waypoints = [
            np.array([300, 15, 13]),
            np.array([300, 55, 13]),
            np.array([300, 95, 13]),
            np.array([300, 135, 13]),
            np.array([300, 175, 13]),
            np.array([300, 215, 13]),
            np.array([300, 255, 13]),
            np.array([300, 295, 13]),
            np.array([300, 335, 13]),
            np.array([285, 15, 13]),
            np.array([285, 55, 13]),
            np.array([285, 95, 13]),
            np.array([285, 135, 13]),
            np.array([285, 175, 13]),
            np.array([285, 215, 13]),
            np.array([285, 255, 13]),
            np.array([285, 295, 13]),
            np.array([285, 335, 13]),
            np.array([45, 15, 13]),
            np.array([45, 55, 13]),
            np.array([45, 95, 13]),
            np.array([45, 135, 13]),
            np.array([45, 175, 13]),
            np.array([45, 215, 13]),
            np.array([45, 255, 13]),
            np.array([45, 295, 13]),
            np.array([45, 335, 13]),
            np.array([30, 15, 13]),
            np.array([30, 55, 13]),
            np.array([30, 95, 13]),
            np.array([30, 135, 13]),
            np.array([30, 175, 13]),
            np.array([30, 215, 13]),
            np.array([30, 255, 13]),
            np.array([30, 295, 13]),
            np.array([30, 335, 13]),

            np.array([65, 35, 13]),
            np.array([65, 75, 13]),
            np.array([65, 115, 13]),
            np.array([65, 155, 13]),
            np.array([65, 195, 13]),
            np.array([65, 235, 13]),
            np.array([65, 275, 13]),
            np.array([65, 315, 13]),
            np.array([80, 35, 13]),
            np.array([80, 75, 13]),
            np.array([80, 115, 13]),
            np.array([80, 155, 13]),
            np.array([80, 195, 13]),
            np.array([80, 235, 13]),
            np.array([80, 275, 13]),
            np.array([80, 315, 13]),
            np.array([250, 35, 13]),
            np.array([250, 75, 13]),
            np.array([250, 115, 13]),
            np.array([2505, 155, 13]),
            np.array([250, 195, 13]),
            np.array([250, 235, 13]),
            np.array([250, 275, 13]),
            np.array([250, 315, 13]),
            np.array([265, 35, 13]),
            np.array([265, 75, 13]),
            np.array([265, 115, 13]),
            np.array([265, 155, 13]),
            np.array([265, 195, 13]),
            np.array([265, 235, 13]),
            np.array([265, 275, 13]),
            np.array([265, 315, 13]),

            np.array([165, 55, 13]),
            np.array([165, 95, 13]),
            np.array([165, 135, 13]),
            np.array([165, 175, 13]),
            np.array([165, 215, 13]),
            np.array([165, 255, 13]),
            np.array([165, 295, 13]),
            np.array([207.5, 175, 13]),
            np.array([122.5, 175, 13]),

            
            # np.array([-3, -2.5, -13]),
            # np.array([-3, 15.3, -13]),
            # np.array([5, 15.3, -13]),
        ]
        self.delta = 3.0
        self.current_leader_target = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _setup_flight(self):
        if self._first_setup:
            self.drone1.reset()
            self.drone2.reset()
            self.drone1.enableApiControl(True, "Drone1")
            self.drone2.enableApiControl(True, "Drone2")
            self.drone1.armDisarm(True, "Drone1")
            self.drone2.armDisarm(True, "Drone2")
            self.drone1.takeoffAsync(vehicle_name="Drone1").join()
            self.drone2.takeoffAsync(vehicle_name="Drone2").join()
            self._first_setup = False
            time.sleep(2)

        self.drone1.moveToPositionAsync(0, 0, -13, 3, vehicle_name="Drone1").join()
        self.drone2.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Drone2").join()
        self.drone1.hoverAsync(vehicle_name="Drone1")
        self.drone2.hoverAsync(vehicle_name="Drone2")
        time.sleep(1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = np.where(img1d <= 1e-3, 1.0, img1d)
        img1d = 255.0 / img1d
        img2d = np.reshape(img1d, (response.height, response.width))
        img2d = cv2.resize(img2d, (self.image_shape[0], self.image_shape[1]))
        return img2d.reshape(self.image_shape) / 255.0

    def _get_obs(self):
        responses = self.drone2.simGetImages(self.image_requests, vehicle_name="Drone2")
        image_front = self.transform_obs(responses[0])
        image_back  = self.transform_obs(responses[1])

        # ✅ 변경: 두 depth 이미지를 병합 (최소값)
        image_merged = np.minimum(image_front, image_back)

        d1 = self.drone1.getMultirotorState(vehicle_name="Drone1").kinematics_estimated
        d2 = self.drone2.getMultirotorState(vehicle_name="Drone2").kinematics_estimated

        dx  = d1.position.x_val - d2.position.x_val
        dy  = d1.position.y_val - d2.position.y_val
        dvx = d1.linear_velocity.x_val - d2.linear_velocity.x_val
        dvy = d1.linear_velocity.y_val - d2.linear_velocity.y_val
        distance = np.linalg.norm([dx, dy])

        state = np.array([dx, dy, dvx, dvy, distance], dtype=np.float32)

        return {"image": image_merged, "state": state}
    

    def _update_obstacle_drones(self):
        for name in self.obstacle_drones:
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            vz = 0
            self.drone2.moveByVelocityAsync(
                vx, vy, vz, duration=self.step_length,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=name
            )


    def _update_leader_movement(self):
        current_pos = self.drone1.simGetObjectPose("Drone1").position
        curr = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])

        if self.current_leader_target is None or np.linalg.norm(curr - self.current_leader_target) < 1.0:
            base = random.choice(self.base_waypoints)
            offset = np.random.uniform(-self.delta, self.delta, size=3)
            target = base + offset
            self.current_leader_target = target
            self.drone1.moveToPositionAsync(
                float(target[0]), float(target[1]), float(target[2]), 1,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name="Drone1"
            )

    def to_quaternion(self, pitch, roll, yaw):
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return Quaternionr(sr * cp * cy - cr * sp * sy,
                           cr * sp * cy + sr * cp * sy,
                           cr * cp * sy - sr * sp * cy,
                           cr * cp * cy + sr * sp * sy)

    def _do_action(self, action):
        self._update_leader_movement()

        vx = float(action[0]) * self.max_forward_speed
        vy = float(action[1]) * self.max_forward_speed
        vz = 0.0  # 고도 유지

        self.drone2.moveByVelocityAsync(
            vx, vy, vz, duration=self.step_length,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name="Drone2"
        ).join()

    def get_distance_to_leader(self):
        d1 = self.drone1.simGetObjectPose("Drone1").position
        d2 = self.drone2.simGetObjectPose("Drone2").position
        return math.sqrt((d1.x_val - d2.x_val)**2 + (d1.y_val - d2.y_val)**2)
    
    def get_min_obstacle_distance(self):
        responses = self.drone2.simGetImages(self.image_requests, vehicle_name="Drone2")
        depths = [np.array(r.image_data_float, dtype=np.float32) for r in responses]
        valid_dists = [np.min(d[d > 1e-3]) for d in depths if np.any(d > 1e-3)]
        return min(valid_dists) if valid_dists else float('inf')

    def is_collision(self):
        collision_info = self.drone2.simGetCollisionInfo(vehicle_name="Drone2")
        return collision_info.has_collided



    def _compute_reward(self, distance, obstacle_dist):
        if distance > 30.0:
            return -1000.0, True

        alpha = 1.0
        # reward_follow = 1.0 if distance <= 10 else math.exp(-alpha * (distance - 10)**2)
        alpha = 0.1
        reward_follow = math.exp(-alpha * distance)
        # reward_follow = 1 / (1 + distance)

        reward_avoid = 0.0
        if obstacle_dist < 2.0:
            k = 1.0
            reward_avoid = -k / (obstacle_dist + 1e-6)

        return reward_follow + reward_avoid, False

    def step(self, action):
        self._do_action(action)
        self._update_obstacle_drones()  # 장애물 드론 계속 움직이게

        obs = self._get_obs()
        distance = self.get_distance_to_leader()
        obstacle_dist = self.get_min_obstacle_distance()

        reward, done_by_reward = self._compute_reward(distance, obstacle_dist)

        collision = self.is_collision()
        if collision:
            reward = -1000.0
            done = True
        else:
            done = done_by_reward or self.step_count >= self.max_steps
        print(f"Step: {self.step_count}, Distance: {distance:.2f}, ObstacleDist: {obstacle_dist:.2f}, Collision: {collision}")

        self.step_count += 1
        return obs, reward, done, {}
    


    def reset(self):
        self.step_count = 0
        self.current_leader_target = None

        self.drone1.simSetVehiclePose(Pose(Vector3r(0, 0, -13), self.to_quaternion(0, 0, 0)), True, "Drone1")
        self.drone2.simSetVehiclePose(Pose(Vector3r(0, 0, -10), self.to_quaternion(0, 0, 0)), True, "Drone2")
        self._update_leader_movement()
                # 장애물 드론 위치 초기화 및 다시 이동 명령 주기
        for name in self.obstacle_drones:
            init_x = random.uniform(0, 300)
            init_y = random.uniform(0, 300)
            self.drone2.simSetVehiclePose(
                Pose(Vector3r(init_x, init_y, -10), self.to_quaternion(0, 0, 0)),
                ignore_collision=True,
                vehicle_name=name
            )

            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            self.drone2.moveByVelocityAsync(
                vx, vy, 0.0, duration=self.step_length,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=name
            )

        return self._get_obs()

    def close(self):
        self.drone1.armDisarm(False, "Drone1")
        self.drone2.armDisarm(False, "Drone2")
        self.drone1.enableApiControl(False, "Drone1")
        self.drone2.enableApiControl(False, "Drone2")
