import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from model.utils.io import write


def _get_OD(file_name):
    try:
        with open(file_name, "rb") as f:
            next(f)
            origin_line = f.readline()
            offset = -50
            while True:
                f.seek(offset, 2)
                lines = f.readlines()
                if len(lines) >= 2:
                    destination_line = lines[-1]
                    break
                offset *= 2
            origin_line = origin_line.decode()
            destination_line = destination_line.decode()
            origin_lst = origin_line.split(",")
            destination_lst = destination_line.split(",")
            od_lst = origin_lst[:2]
            od_lst.append(destination_lst[1])
            od_lst += origin_lst[2:4] + destination_lst[2:4]
            return od_lst
    except Exception as err:
        print(err)


def get_OD(path, file_name):
    user_ods = []
    for filename in file_name:
        file_path = os.path.join(path, filename)
        od = _get_OD(file_path)
        user_ods.append(od)
    od_df = pd.DataFrame(data=user_ods,
                         columns=["userid", "origin_time", "destination_time", "origin_lat", "origin_lon",
                                  "destination_lat", "destination_lon"])
    return od_df


def get_FP(od, stay_time=pd.Timedelta('0 days 00:20:00')):
    try:
        sp = pd.DataFrame()
        sp["userid"] = od["userid"]
        sp["checkin_time"] = od["destination_time"]
        sp["checkout_time"] = od["origin_time"].shift(-1)
        sp["sp_lat"] = od["destination_lat"]
        sp["sp_lon"] = od["destination_lon"]
        sp.drop([len(sp) - 1], inplace=True)
        sp["checkin_time"] = pd.to_datetime(sp["checkin_time"])
        sp["checkout_time"] = pd.to_datetime(sp["checkout_time"])
        sp["stay_duration"] = sp["checkout_time"] - sp["checkin_time"]
        sp = sp[sp["stay_duration"] >= stay_time]
        # [0-6] corresponds to Monday to Sunday
        sp["sp_week_day"] = sp["checkout_time"].dt.dayofweek

        # Modify data column type
        sp["sp_lat"] = pd.to_numeric(sp["sp_lat"])
        sp["sp_lon"] = pd.to_numeric(sp["sp_lon"])
        return sp
    except Exception as err:
        print(err)


def cluster(df, method="DBSCAN"):
    coords = df[["sp_lat", "sp_lon"]].values
    if method == "DBSCAN":
        kms_per_radian = 6371.0088
        epsilon = 0.2 / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        cluster_labels = db.labels_
        # -1 in cluster_label represents noise point
        df["cluster_label"] = cluster_labels
        # number of clusters, noise points are not considered to be a cluster
        num = len(set([n for n in cluster_labels if n >= 0]))
        clusters = pd.Series([coords[cluster_labels == n] for n in range(num)], dtype='object')
        center_points = clusters[:].map(get_centermost_point)
        return df, num - 1, center_points


def clean_cluster(df, center_points):
    df["cluster_lat"] = df.apply(
        lambda row: row["sp_lat"] if row["cluster_label"] == -1 else center_points[row["cluster_label"]][0], axis=1)
    df["cluster_lon"] = df.apply(
        lambda row: row["sp_lon"] if row["cluster_label"] == -1 else center_points[row["cluster_label"]][1], axis=1)
    return df


def get_centermost_point(cluster, method="get_centroid"):
    cluster = np.array(cluster)
    centroid = cluster.mean(axis=0)
    if method == "get_centroid":
        return tuple(centroid)
    elif method == "get_center_point":
        centermost_point = min(cluster, key=lambda point: great_circle(point[::-1], centroid).m)
        return tuple(centermost_point)


def generate_L(INPUT_PATH=r'./result/trips', OUTPUT_PATH=r'./result/L'):
    """
     cluster flameout points into stay points and form the stay point sequence L.
    :param INPUT_PATH: str, optional
        The root folder where the trips is located.
    :param OUTPUT_PATH: str, optional
        The folder where the generated Ls are stored.
    :return:
    """
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    for root, dirs, files in os.walk(INPUT_PATH):
        if files:
            userid = files[0].split(" ")[0]
            print(userid)
            od_df = get_OD(root, files)
            sp_df = get_FP(od_df)
            if len(sp_df) < 1:
                print('This individual has no stay point')
                continue
            df_cluster, num_cluster, center_most_point = cluster(sp_df)
            df_cleaned_cluster = clean_cluster(df_cluster, center_most_point)
            output_csv_name = os.path.join(OUTPUT_PATH, "L_" + str(userid) + ".csv")
            write(output_csv_name, df_cleaned_cluster)


if __name__ == '__main__':
    input_path = r'../../result/trips'
    output_path = r'../../result/L'
    generate_L(input_path, output_path)
