from dotenv import dotenv_values

default_envs = {"camera": 0, "canny-min": 70, "canny-max": 100, "canny-aperture": 5, "approx-epsilon": 0.04}
config = {
    **default_envs,
    **dotenv_values(".env")
}
