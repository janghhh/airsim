import subprocess
import time
import sys
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os

import os
print("🔧 장애물 드론 실행 중...")
subprocess.Popen(
    ["start", "", "cmd", "/k", "C:\\Users\\jangh\\Documents\\AirSim\\PythonClient\\multirotor\\run_add_drone.bat"],
    shell=True
)

# 💡 드론이 흩어질 시간 확보
time.sleep(10)

# ✅ 이제 강화학습 관련 코드 실행
print("✅ 2단계: 강화학습 시작")
sys.path.append(r"C:\Users\jangh\Documents\AirSim\PythonClient\reinforcement_learning")

# 환경 import
from airgym.envs.drone_follower_env import DroneFollowerEnv

# 로그 기록 콜백 클래스 정의
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        reward = self.locals['rewards'][0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            print(f"Episode {len(self.episode_rewards)+1}:")
            print(f"  Total Reward: {self.current_episode_reward:.2f}")
            print(f"  Length: {self.current_episode_length} steps")

            self.logger.record("train/episode_reward", self.current_episode_reward)
            self.logger.record("train/episode_length", self.current_episode_length)

            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

# 환경 생성 함수
def create_env():
    return DroneFollowerEnv(
        ip_address="127.0.0.1",
        step_length=0.25,
        image_shape=(84, 84, 1),
        max_steps=500,
        leader_behavior="random"
    )

# PPO 학습 구성
vec_env = make_vec_env(create_env, n_envs=1)
episode_logger = EpisodeLoggerCallback()

model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./ppo_logs/",
    device='cuda',
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

model.learn(
    total_timesteps=5_000_000,
    callback=episode_logger,
    tb_log_name="drone_follower",
    progress_bar=True
)

model.save("drone_follower_ppo")
vec_env.close()
