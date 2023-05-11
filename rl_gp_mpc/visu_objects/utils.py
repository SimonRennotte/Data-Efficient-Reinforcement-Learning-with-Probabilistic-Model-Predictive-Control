import datetime
import os


def get_env_name(env):
    try:
        env_name = env.env.spec.entry_point.replace('-', '_').replace(':', '_').replace('.', '_')
    except:
        env_name = env.name
    return env_name


def create_folder_save(env_name):
	datetime_now = datetime.datetime.now()
		
	folder_save = os.path.join('folder_save', env_name, datetime_now.strftime("%Y_%m_%d_%H_%M_%S"))
	if not os.path.exists(folder_save):
		os.makedirs(folder_save)
	return folder_save
