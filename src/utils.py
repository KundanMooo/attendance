def log_event(message):
    with open("app.log", "a") as log_file:
        log_file.write(f"{message}\n")

def read_settings(file_path):
    settings = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split('=')
            settings[key] = value
    return settings

def get_save_path(base_path, filename):
    import os
    return os.path.join(base_path, filename)