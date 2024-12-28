from dotenv import dotenv_values

default_envs = {"camera": 0, "canny-min": 70, "canny-max": 100, "canny-aperture": 5, "approx-epsilon": 0.02, "min-area":1000 }
config = {
    **default_envs,
    **dotenv_values(".env")
}

# To get value of selected size of paper
selected_option = None
