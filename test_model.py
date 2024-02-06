# importer the game
import gym_super_mario_bros
# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import PPO for algos
from stable_baselines3 import PPO
# import frame stacker wrapper and grayscaling wrapper
from gym.wrappers import FrameStack, GrayScaleObservation
# import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

MARIO_ENV = "SuperMarioBros-v0"
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# 1. create the base environment
def create_env():
    # 2. Simplify the controls
    return JoypadSpace(gym_super_mario_bros.make(MARIO_ENV),
                        SIMPLE_MOVEMENT)

def preprocess(env):
    # 3. GrayScale
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env]) # type: ignore
    # 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')
    return env

env = preprocess(create_env)
model = PPO.load('./train/best_model_5000')

state = env.reset()
while True:
    action, state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()