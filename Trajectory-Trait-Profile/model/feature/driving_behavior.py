import os
import shutil
import time
from multiprocessing import Pool
import pandas as pd
from math import pi
import geopandas as gpd
import datetime
from model.utils.io import write
from model.feature.feature import read_L
from itertools import repeat


def csv2shp(csv_fname, lon, lat):
    """
    Convert csv file to shp format.
    :param csv_fname: str
        The name of the csv file.
    :param lon: str
        Column name as longitude in csv file.
    :param lat: str
        Column name as latitude in csv file.
    :return: GeoDataFrame
        Trajectory in shp format.
    """
    csv_df = pd.read_csv(csv_fname, dtype={lon: float, lat: float})
    shp_gdf = gpd.GeoDataFrame(csv_df, geometry=gpd.points_from_xy(csv_df[lon],
                                                                   csv_df[lat],
                                                                   crs="EPSG:4326"))
    return shp_gdf


def harsh_acceleration(data, threshold=1.67):
    """
    Judgment of harsh acceleration, if the acceleration is greater than the threshold, it is judged as harsh acceleration.
    :param data: series
        Acceleration.
    :param threshold: float
        The threshold for judging harsh acceleration.
    :return: int
        Number of trajectory points have harsh acceleration.
    """
    sum_harsh_acceletation = data[data > threshold].count()
    return sum_harsh_acceletation


def harsh_breaking(data, threshold=-1.67):
    """
    Judgment of harsh breaking, if the acceleration is lesser than the threshold, it is judged as harsh breaking.
    :param data: series
        Acceleration.
    :param threshold: float
        The threshold for judging harsh breaking.
    :return: int
        Number of trajectory points have harsh breaking.
    """
    sum_harsh_breaking = data[data < threshold].count()
    return sum_harsh_breaking


def harsh_steering(data, threshold=0.5):
    """
    Judgment of harsh steering, if the turning angle is greater than the threshold, it is judged as harsh steering.
    :param data: series
        Turning angle.
    :param threshold: float
        The threshold for judging harsh steering.
    :return: int
        Number of trajectory points have harsh steering.
    """
    data_a_1 = data["direction"].shift()
    data_t_1 = data["checkin_time"].shift()
    angle = (data_a_1 - data["direction"]) / 180 * pi
    time_delta = (data["checkin_time"] - data_t_1).dt.seconds
    steer = abs(angle.values) / time_delta
    sum_harsh_steering = steer[steer > threshold].count()
    return sum_harsh_steering


def over_speed(data, threshold=(80 / 3.6)):
    """
    Judgment of speeding, if the speed is greater than the threshold, it is judged as speeding.
    :param data: series
        Speed.
    :param threshold: float
        The threshold for judging speeding.
    :return: int
        Number of trajectory points are speeding.
    """
    sum_over_speed = data[data > threshold].count()
    return sum_over_speed


def speed_acceleration_describe(traj):
    """
    Statistical driving characteristics.
    :param traj: dataframe
        Trajectory.
    :return: tuple
        Including average speed, the standard deviation of the speed, the standard deviation of the speed, the standard deviation of the speed acceleration.
    """
    speed_mean = traj["speed"].mean()
    speed_std = traj["speed"].std()
    speed_max = traj['speed'].max()
    a_std = traj["acceleration"].std()
    return speed_mean, speed_std, speed_max, a_std


def _create_junction_buffer(junction, distance=10):
    """
    Create the buffers for intersections
    :param junction: GeoDataFrame
        The intersection data in format shp.
    :param distance: int, optional
        The radius of buffers.(m)
    :return: GeoDataFrame
        The buffers of intersections.
    """
    junction['geometry'] = junction['geometry'].to_crs(32649)
    junction_buffer = junction['geometry'].buffer(distance).to_crs(4326)
    junction_buffer = gpd.GeoDataFrame(junction_buffer, columns=['geometry'])
    entry_directory = os.path.dirname(os.path.abspath(__file__))
    # junction_buffer.to_file(r'./auxiliary_data/buffer_junction.geojson', driver='GeoJSON')
    junction_buffer.to_file(r'./model/feature/auxiliary_data/buffer_junction.geojson', driver='GeoJSON')
    # print('create_junction_buffer is ok')
    return junction_buffer


def junction_speed(junction_buffer, traj):
    """
    Computer intersection’s ratio of speeding points and intersection’s average speed of a trajectory.
    :param junction_buffer: GeoDataFrame
        The buffers of intersections.
    :param traj: GeoDataFrame
        A trajectory of the individual.
    :return: tuple
        Intersection’s average ratio of speeding points and intersection’s average speed
    """
    threshold = 30 / 3.6  # m/s
    junction_sjoin_traj = gpd.sjoin(traj, junction_buffer, predicate='within')
    if len(junction_sjoin_traj) > 0:
        over_speed = len(junction_sjoin_traj[junction_sjoin_traj['speed'] > threshold]) / len(junction_sjoin_traj)
        mean_speed = junction_sjoin_traj['speed'].mean()
        return over_speed, mean_speed
    else:
        return 0, 0


def fatigue_driving(traj, threshold=datetime.timedelta(hours=4)):
    """
    Determine whether fatigue driving. Continuous driving time is larger than the threshold, it is determined as fatigue driving.
    :param traj: GeoDataFrame
        A trajectory of the individual.
    :param threshold: timedelta, optional
        The threshold for judging fatigue driving.
    :return: boolean
        1 represent this trip is fatigue_driving.
    """
    if (traj.iat[-1, 1] - traj.iat[0, 1]) > threshold:
        return 1
    else:
        return 0


def driving_behavior_trip(intersection_buffer, traj):
    """
    List the features of a trip.
    :param intersection_buffer: GeoDataFrame
        the buffer of intersection.
    :param traj: GeoDataFrame
        A trajectory of the individual.
    :return: list
        The features of a trip.
    """
    points_num = traj.shape[0]
    ha_points = harsh_acceleration(traj["acceleration"])
    hb_points = harsh_breaking(traj["acceleration"])
    hs_points = harsh_steering(traj[["checkin_time", "direction"]])
    ha_points_ratio = ha_points / points_num
    hb_points_ratio = hb_points / points_num
    hs_points_ratio = hs_points / points_num

    os_points = over_speed(traj["speed"])
    os_points_ratio = os_points / points_num

    speed_m, speed_s, speed_max, acceleration_s = speed_acceleration_describe(traj)

    junction_over_speed, junction_speed_mean = junction_speed(intersection_buffer, traj)

    fatigue_drive = fatigue_driving(traj)

    a_traj_feature = [traj.iat[0, 0], traj.iat[0, 1], traj.iat[-1, 1], speed_m, speed_s, speed_max, acceleration_s,
                      ha_points,
                      hb_points, hs_points, ha_points_ratio, hb_points_ratio, hs_points_ratio, os_points,
                      os_points_ratio,
                      junction_over_speed, junction_speed_mean, fatigue_drive]
    return a_traj_feature


def driving_behavior_individual(files_path, intersection_buffer, L_PATH, OUTPUT_PATH):
    """
    calculate individual driving behavior on each trip
    :param file_path: the path of individual trips
    :param L_PATH:  the path of Ls
    :param OUTPUT_PATH: the path of output
    :return: none
    """
    files = os.listdir(files_path)
    userid = files[0].split(" ")[0]
    output_file_name = userid + "_L_with_driving_behavior.csv"
    if os.path.exists(os.path.join(OUTPUT_PATH, output_file_name)):
        print('The driving behavior features of the individual have been calculated')
        return None

    # calculate driving behavior on each trip
    harsh_all_traj = []
    for file_name in files:
        file_name = os.path.join(files_path, file_name)
        traj = csv2shp(file_name, 'lon', 'lat')
        traj['checkin_time'] = pd.to_datetime(traj['checkin_time'], format='%Y-%m-%d %H:%M:%S')
        harsh = driving_behavior_trip(intersection_buffer, traj)
        harsh_all_traj.append(harsh)
    OD_with_driving_behavior = pd.DataFrame(data=harsh_all_traj,
                                            columns=["userid", "checkout_time", "checkin_time", "speed_mean",
                                                     "speed_std",
                                                     "speed_max",
                                                     "acceleration_std", "harsh_acceleration",
                                                     "harsh_breaking",
                                                     "harsh_steering", "harsh_acceleration_ratio",
                                                     "harsh_breaking_ratio",
                                                     "harsh_steering_ratio", "over_speed", "over_speed_ratio",
                                                     "junction_over_speed", "junction_speed_mean",
                                                     "fatigue_driving"])

    # merging driving behavior features onto L
    L = read_L(os.path.join(L_PATH, "L_" + str(userid) + ".csv"))
    OD_with_driving_behavior.drop(columns=['userid', 'checkout_time'], inplace=True)
    L_with_driving_behavior = pd.merge(L, OD_with_driving_behavior, on='checkin_time')

    output_file_path = os.path.join(OUTPUT_PATH, output_file_name)
    write(output_file_path, L_with_driving_behavior)


def driving_behavior(TRIPS_PATH=r"./result/trips", ROAD_PATH=r'./model/feature/auxiliary_data', L_PATH=r'./result/L',
                     OUTPUT_PATH=r"./result/L_with_driving_behavior"):
    """
     Extract driving behavior features and merge them with L.
    :param TRIPS_PATH: str, optional
        The root folder where the generated trips are stored.
    :param ROAD_PATH: str, optional
        The folder where the road intersection data is stored.
    :param L_PATH: str, optional
        The folder where the generated Ls are stored.
    :param OUTPUT_PATH: str, optional
         The folder where the L with driving behavior features are stored.
    """
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    junction = csv2shp(os.path.join(ROAD_PATH, 'selected_node.csv'), 'x_coord', 'y_coord')
    intersection_buffer = _create_junction_buffer(junction)
    # Read the trajectory data, and judge the harsh acceleration, harsh deceleration, and harsh steering
    # (use multiprocessing to speed up)
    individual_dirs = os.listdir(TRIPS_PATH)
    individual_paths = [os.path.join(TRIPS_PATH, individual_dir) for individual_dir in individual_dirs]
    with Pool(processes=os.cpu_count()) as pool:
        for individual_path in individual_paths:
            pool.apply_async(driving_behavior_individual,
                             args=(individual_path, intersection_buffer, L_PATH, OUTPUT_PATH))
        pool.close()
        pool.join()


if __name__ == '__main__':
    trips_path = r"../../result/trips"
    l_path = r'../../result/L'
    output_path = r"../../result/L_with_driving_behavior"
    road_path = r'./auxiliary_data'
    start_time = time.time()
    driving_behavior(trips_path, road_path, l_path, output_path)
    end_time = time.time()
    print('execution time:{}'.format(end_time - start_time))
