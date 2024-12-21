from dotenv import dotenv_values

default_envs = {"camera": 0, "canny-min": 50, "canny-max": 100, "canny-aperture": 5, "approx-epsilon": 0.02}
config = {
    **default_envs,
    **dotenv_values(".env")
}
