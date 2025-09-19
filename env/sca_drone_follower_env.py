from collections import deque
import setup_path
import airsim
import random
import numpy as np
import math
import time
from airsim import Vector3r, Pose, Quaternionr
from argparse import ArgumentParser
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class AirSimDroneEnv(AirSimEnv):
    """
    ----------------------------------------------------------------------------------------------------------------------------
    __init__: 초기화 및 관측, 액션 공간 설정 함수. 수렴 관련 파라미터는 제거할 예정입니다.
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def __init__(self, ip_address, step_length = 1.0):
        super().__init__()
        self.step_length = step_length        
        self.leader_velocity = 16
        # global 변수처럼 활용할 state 값들. 실제 state의 기능을 하는 것은 아니다.
        self.state = {
            "relative_position": np.zeros(2), 
            "collision": False,
            "min_distances": np.zeros(36)
        }
        # lidar_dim = 5        # Lidar 방향 섹터 값
        # leader_info_dim = 2  # 유인기 상대 위치 정보
        # 실제 state 기능을 하는 observation_space의 명세표.
        self.observation_space = spaces.Box(
            low=0.0,
            high=20.0,
            shape=(39,),  # 36 lidar + 1 distance + 2 relative pos
            dtype=np.float32
        )

        #airsim 시뮬레이터에 접근하게 해주는 API 
        self.drone = airsim.MultirotorClient(ip=ip_address) # 그래서 따로 무인기 유인기 정의할 필요없이 그냥 drone 이름 사용하면 된다.

        #self.action_space = spaces.Discrete(7) #얘는 일단 나중에 연속 행동으로 바꿔야한다.
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self.fixed_z = -10
        self.leader_speed = 1.0
        self.step_duration = 0.01
        self.steps_to_next_waypoint = 0
        self.total_steps = 0
        self.episode_number = 0
        self.episode_reward = 0.0
        self.leader_waypoints = [] # 유인기가 움직일 경로 초기화 
        self._first_setup = True
        self._setup_flight()

    """
    ----------------------------------------------------------------------------------------------------------------------------
    _generate_leader_waypoints: 유인기 행동 경로 생성 함수. 
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def _generate_leader_waypoints(self):
        # 맵 4개로 쪼갠 중심 위치
        centers = [
            # 수정
            np.array([25, 25, self.fixed_z]),
            np.array([-35, 25, self.fixed_z]),
            np.array([-35, -30, self.fixed_z]),
            np.array([25, -30, self.fixed_z])   
        ]

        self.leader_waypoints = []
        for _ in range(5):
            center = random.choice(centers)
            offset_x = np.round(np.random.uniform(-35, 35), 2)  # 유인기가 가장자리까지 돌되 맵에서 벗어나지 않도록 해보았어요
            offset_y = np.round(np.random.uniform(-25, 25), 2)
            waypoint = [center[0] + offset_x, center[1] + offset_y, self.fixed_z]
            self.leader_waypoints.append(waypoint)
            


    """----------------------------------------------------------------------------------------------------------------------------
    _update_leader_movement: 유인기가 경로대로 움직이도록 하는 함수. 무한 루프를 돌 수 있도록 수정할 예정입니다. 
                             또한, 첫번째 안으로 moveOnPath 대신 airsim.Pose를 사용하여 무인기와 함께 텔레포트를 적용해볼 예정입니다.
    -------------------------------------------------------------------------------------------------------------------------------
    """
    def _update_leader_movement(self):
        """
        리더 드론을 생성된 웨이포인트에 따라 이동시킵니다.
        목표 지점에 도착하면 다음 웨이포인트로 목표를 변경합니다.
        """
        # 따라갈 웨이포인트가 없거나 모든 경로를 통과했으면 함수를 종료합니다.
        if not hasattr(self, 'leader_waypoints') or not self.leader_waypoints or self.current_waypoint_idx >= len(self.leader_waypoints):
            self._generate_leader_waypoints()
            self.current_waypoint_idx = 0

        # 1. 현재 목표 웨이포인트와 드론의 현재 위치를 가져옵니다.
        target_pos_np = np.array(self.leader_waypoints[self.current_waypoint_idx])
        
        current_pose = self.drone.simGetVehiclePose(vehicle_name="Drone1")
        current_pos_np = np.array([
            current_pose.position.x_val,
            current_pose.position.y_val, 
            current_pose.position.z_val
        ])

        # 2. 목표 지점 도착 여부를 확인합니다. (Z축은 무시하고 XY 평면 거리만 계산)
        dist_to_target = np.linalg.norm(target_pos_np[:2] - current_pos_np[:2])
        arrival_threshold = 2.0  # 2미터 이내로 들어오면 도착으로 간주

        if dist_to_target < arrival_threshold:
            self.current_waypoint_idx += 1
            
            # 모든 웨이포인트를 통과했는지 확인
            if self.current_waypoint_idx >= len(self.leader_waypoints):
                # 모든 경로 완주 후 동작 (예: 순찰 중지, 새 웨이포인트 생성 등)
                return

            # 목표가 바뀌었으므로, 다음 로직에서 새 목표를 사용하도록 target_pos_np를 다시 설정합니다.
            target_pos_np = np.array(self.leader_waypoints[self.current_waypoint_idx])

        # 3. 목표 방향으로 이동할 'offset'을 계산합니다.
        # 목표 지점을 향하는 방향 벡터를 계산합니다.
        direction_vector = target_pos_np - current_pos_np
        distance = np.linalg.norm(direction_vector)
        
        if distance < 1e-6: # 거리가 매우 가까우면 이동 계산을 건너뜁니다.
            return
            
        # 방향 벡터를 정규화(크기를 1로 만듦)하여 순수 방향만 얻습니다.
        unit_direction = direction_vector / distance

        # 한 번에 이동할 거리(step)입니다. action space와 유사하게 0.3으로 설정했습니다.
        step_size = 0.3
        
        # 이동 거리는 step_size와 목표까지 남은 거리 중 작은 값으로 정합니다. (오버슈팅 방지)
        move_distance = min(step_size, distance)
        
        # 최종 이동 벡터(offset) = 방향 * 거리
        offset_vector = unit_direction * move_distance

        # 4. 드론의 새로운 위치를 계산하고 적용합니다.
        new_pos_np = current_pos_np + offset_vector

        new_position_airsim = airsim.Vector3r(new_pos_np[0], new_pos_np[1], new_pos_np[2])
        level_quaternion = airsim.Quaternionr(0, 0, 0, 1) # 드론 수평 유지
        new_pose = airsim.Pose(new_position_airsim, level_quaternion)
        
        self.drone.simSetVehiclePose(new_pose, ignore_collision=False, vehicle_name="Drone1")

    """
    ----------------------------------------------------------------------------------------------------------------------------
    close : 학습 종료 시 호출되는 함수
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def close(self):
        super().close()
        try:
            if self.drone:
                self.drone.reset()
                self.drone.enableApiControl(False, vehicle_name="Drone1")
                self.drone.enableApiControl(False, vehicle_name="Drone2")
        except Exception as e:
            print(f"Error during AirSimDroneEnv close: {e}")

      
    """
    ----------------------------------------------------------------------------------------------------------------------------
    _setup_flight: 학습 시작 시 드론 활성화 및 이륙 함수
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def _setup_flight(self):
        if self._first_setup:
            self.drone.reset()
            #드론의 API 활성화
            self.drone.enableApiControl(True,vehicle_name = "Drone1")
            self.drone.enableApiControl(True,vehicle_name = "Drone2")

            #드론 시동걸기
            self.drone.armDisarm(True,vehicle_name = "Drone1")  
            self.drone.armDisarm(True,vehicle_name = "Drone2")

            # 이륙
            self.drone.takeoffAsync(vehicle_name="Drone1")
            self.drone.takeoffAsync(vehicle_name="Drone2").join()
            time.sleep(1)
            self._first_setup = False

        
        # self.drone.hoverAsync(vehicle_name="Drone1")
        # self.drone.hoverAsync(vehicle_name="Drone2")

        self.drone.moveToPositionAsync(5, 0, self.fixed_z, 3, vehicle_name="Drone1")
        self.drone.moveToPositionAsync(0, 0, self.fixed_z, 3, vehicle_name="Drone2").join()
        time.sleep(0.1)

    """
    ----------------------------------------------------------------------------------------------------------------------------
    _get_obs : observation space에 값을 넣어주는 함수 
    ----------------------------------------------------------------------------------------------------------------------------
    """
    # 라이더 센서 값 중 최소 거리들 + 유인기 상대 위치
    def _get_obs(self):

        # 2. 충돌
        collision = self.drone.simGetCollisionInfo(vehicle_name="Drone2").has_collided
        self.state["collision"] = collision

        # 3. 위치 및 거리 계산
        D1_pos = self.drone.simGetObjectPose("Drone1").position
        D2_pos = self.drone.simGetObjectPose("Drone2").position
        dist = math.sqrt((D1_pos.x_val - D2_pos.x_val) ** 2 + (D1_pos.y_val - D2_pos.y_val) ** 2)
        self.state["dist"] = dist
        relative_pos = np.array([D1_pos.x_val - D2_pos.x_val, D1_pos.y_val - D2_pos.y_val], dtype=np.float32)

        # 4. Lidar
        Lidar = self.drone.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone2")
        points = np.array(Lidar.point_cloud, dtype=np.float32).reshape(-1, 3)
        angles = np.arctan2(points[:, 1], points[:, 0])
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        num_bins = 36
        bin_indices = ((angles + np.pi) / (2 * np.pi) * num_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        min_distances = np.full(num_bins, 20.0, dtype=np.float32)
        for i in range(len(points)):
            bin_idx = bin_indices[i]
            min_distances[bin_idx] = min(min_distances[bin_idx], distances[i])

        self.state["min_distances"] = min_distances

        # 5. 관측 조합 (이제야 가능!)
        clipped_dist = np.clip(dist, 0.0, 30.0)
        obs = np.concatenate([min_distances, [clipped_dist], relative_pos]).astype(np.float32)
        return obs
 
    
    """
    ----------------------------------------------------------------------------------------------------------------------------
    _do_action : 에이전트의 행동 알고리즘 함수. 두번째 안으로 이동 api를 moveByVelocityAsync를 사용하도록 수정할 예정입니다.
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def _do_action(self, action):
        pose = self.drone.simGetVehiclePose("Drone2")

        x = pose.position.x_val
        y = pose.position.y_val
        z = self.fixed_z

        x_quad_offset = float(action[0]) * self.step_length 
        y_quad_offset = float(action[1]) * self.step_length 
        new_position = airsim.Vector3r(x+x_quad_offset,y+y_quad_offset,z)

        # 원하는 오일러 각 (라디안)
        # 예를 들어, 피치 0.1 라디안 (약 5.7도), 롤 0, 요 0.1 라디안 (약 5.7도)
        _pitch = 0.0
        _roll = 0.0
        _yaw = 0.0
        _orientation_quaternion = airsim.to_quaternion(_pitch, _roll, _yaw) #굳이 사용자 함수 만들 필요없이 airsim api 사용하면 되네 
        new_pose = airsim.Pose(new_position, _orientation_quaternion)
        self.drone.simSetVehiclePose(new_pose, False, vehicle_name="Drone2")
        # 설마 이거 true로 해서...?
   
    """
    ----------------------------------------------------------------------------------------------------------------------------
    _compute_reward : 보상 로직 함수. 
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def _compute_reward(self):
        done = 0
        reward = 0

        # --- 1단계: 생존 확인 (가장 중요) ---
        # 충돌 시 큰 페널티를 주고 즉시 에피소드 종료
        if self.state["collision"]:
            reward = -100.0  # 생존이 중요하므로 페널티를 더 강화
            done = 1
            return reward, done

        dist_to_leader = self.state["dist"]

        # 원거리 이탈 시에도 페널티
        if dist_to_leader > 60.0:
            reward = -20.0
            done = 1
            return reward, done

        # --- 2단계: 안전 거리 확보 보상 ---
        # 가장 가까운 장애물과의 거리를 기준으로 '안전 보상'을 계산
        # 이 보상은 항상 0보다 크거나 같음
        min_dist_to_obstacle = np.min(self.state["min_distances"])
        
        # 예: 10m 이상 떨어져 있으면 최대 보너스 5점
        # 가까워질수록 보너스가 0에 가까워짐
        SAFETY_BONUS_CAP = 5.0
        safety_reward = min( (min_dist_to_obstacle / 10.0) * SAFETY_BONUS_CAP, SAFETY_BONUS_CAP)

        # --- 3단계: 추적 성공 보상 ---
        # 최적 거리(10m)에 가까울수록 높은 보너스를 주는 '추적 보상'
        OPTIMAL_DISTANCE = 10.0
        dist_err = abs(dist_to_leader - OPTIMAL_DISTANCE)
        
        # 보상 감소 속도를 조절 (k값이 클수록 빠르게 감소)
        k = 0.15 
        tracking_reward = 10.0 * math.exp(-k * dist_err)

        # --- 최종 보상: 기본 안전 보상 + 추적 성공 보너스 ---
        # 에이전트는 기본적으로 안전 보상을 받으려 노력하고,
        # 더 높은 점수를 위해 추적 보너스를 노리게 됨
        reward = safety_reward + tracking_reward
        
        return reward, done


    def visualize(self):
        center_pos = self.drone.simGetObjectPose("Drone1").position
        follower_pos = self.drone.simGetObjectPose("Drone2").position
        radius = 10.0
        points = []
        radius_red = 60.0
        points_red = []

        for i in range(37): 
            angle = (i / 36) * 2 * np.pi  # 0 to 2*pi
            x = center_pos.x_val + radius * np.cos(angle)
            y = center_pos.y_val + radius * np.sin(angle)
            z = center_pos.z_val # 드론과 같은 높이에 그림
            points.append(airsim.Vector3r(x, y, z))

            x = center_pos.x_val + radius_red * np.cos(angle)
            y = center_pos.y_val + radius_red * np.sin(angle)            
            points_red.append(airsim.Vector3r(x, y, z))

        self.drone.simPlotLineStrip(
            points=points,
            color_rgba=[1.0, 1.0, 0.0, 0.8],
            thickness=10.0,
            duration=0.1,
            is_persistent=False
        )
        self.drone.simPlotLineStrip(
            points=points_red,
            color_rgba=[1.0, 0.0, 0.0, 0.8], 
            thickness=10.0,
            duration=0.1,
            is_persistent=False
        )

        self.drone.simPrintLogMessage("Drone1", "LEADER", 1)  # 일종의 콘솔 로그
        self.drone.simPlotPoints([airsim.Vector3r(center_pos.x_val, center_pos.y_val, center_pos.z_val + 2.0)], color_rgba=[1,0,0,1], size=15)

        self.drone.simPrintLogMessage("Drone2", "FOLLOWER", 1)  # 일종의 콘솔 로그
        self.drone.simPlotPoints([airsim.Vector3r(follower_pos.x_val, follower_pos.y_val, follower_pos.z_val + 2.0)], color_rgba=[0,1,0,1], size=15)

    """
    ----------------------------------------------------------------------------------------------------------------------------
    step : 매 스텝마다 호출되는 함수.
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def step(self, action):
        self._do_action(action)
        self._update_leader_movement()
        self.visualize()

        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.current_step += 1
        self.total_steps += 1
        self.episode_reward += reward

        #if self.episode_reward >= 5000:
        #    done = 1

        print(f"step: {self.current_step}, reward : {reward}")

        return obs, reward, done, self.state

    """
    ----------------------------------------------------------------------------------------------------------------------------
    reset : 한 에피소드 종료 시 호출되는 함수. 수렴 여부 확인 로직을 삭제할 예정입니다.
    ----------------------------------------------------------------------------------------------------------------------------
    """
    def reset(self):
        self.current_step = 0
        self.step_count = 0
        self.leader_waypoints = []

        
        if self.total_steps > 0:
            print("--------------------------------------------------------------------------------------------------------")
            print(f"Episode {self.episode_number}, Total Reward: {self.episode_reward}, Total Steps: {self.total_steps}")
            print("--------------------------------------------------------------------------------------------------------")
            
        
        self.episode_number += 1
        self.episode_reward = 0.0
        self.current_step = 0

        self.drone.simSetVehiclePose(airsim.Pose(airsim.Vector3r(5, 0, self.fixed_z)), True, "Drone1")
        self.drone.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z)), True, "Drone2")
        # 유인기 웨이포인트 초기화 및 재생성

        self._generate_leader_waypoints() # 새로운 웨이포인트 생성
        self.current_waypoint_idx = 0
        self.state = { "relative_position": np.zeros(2), "collision": False, "min_distances": np.zeros(36)}
        return self._get_obs()
