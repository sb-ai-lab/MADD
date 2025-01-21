import socket

import yaml


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("10.254.254.254", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


file_name = "multi_agents_system/config.yaml"
data = open(file_name, "r")
config = yaml.load(data, Loader=yaml.Loader)

config["frontend_address"] = get_ip()
config["models_address"] = "127.0.0.1"  # get_ip()


with open(file_name, "w") as yaml_file:
    yaml_file.write(yaml.dump(config, default_flow_style=False))
