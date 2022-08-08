#
import os
import shutil
import pandas as pd
import datetime
from model.utils.io import write


def read_traj(file_name):
    data = pd.read_csv(file_name)
    data["checkin_time"] = pd.to_datetime(data["checkin_time"])
    return data


def need_concat(traj, traj_next, time_threshold=datetime.timedelta(minutes=20)):
    """

    :param traj:
    :param traj_next:
    :param time_threshold:
    :return:
    """
    if (traj_next.iat[0, 1] - traj.iat[-1, 1]) > time_threshold:
        return False
    else:
        # print('need merge')
        return True


def concat_traj(input_root, output_root):
    for root, dirs, files in os.walk(input_root):
        if files:
            output_filename = files[0]
            userid = output_filename.split(" ")[0]
            output_dir = output_root + "./" + userid

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)

            traj = read_traj(os.path.join(root, output_filename))
            files.pop(0)
            for filename in files:
                print(filename)
                traj_next = read_traj(os.path.join(root, filename))
                if need_concat(traj, traj_next):
                    traj = pd.concat([traj, traj_next], ignore_index=True)
                else:
                    fname = os.path.join(output_root, userid, output_filename)
                    write(fname, traj)
                    traj = traj_next
                    output_filename = filename
            write(os.path.join(output_root, userid, output_filename), traj)


def generate_trips(INPUT_PATH=r'./data/sample_trajectory', OUTPUT_PATH=r'./result/trips'):
    """
    Concat trajectories with stay time less than the threshold into trips.
    :param INPUT_PATH: str, optional
        The root folder where the original trajectory is located. The root can contain multiple folders, one individual corresponds to one folder. Individual trajectories are organized in multiple csv.
    :param OUTPUT_PATH: str, optional
        The root folder where the generated trips are stored.
    """
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)
    concat_traj(INPUT_PATH, OUTPUT_PATH)


if __name__ == '__main__':
    input_path = r'../../data/sample_trajectory'
    output_path = r'../../result/trips'
    generate_trips(input_path, output_path)
