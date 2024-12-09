# The constant name used in the program, if the attribute name is changed, only need to change this file

USER = "userid"
CHECK_IN_TIME = "checkin_time"
CHECK_OUT_TIME = "checkout_time"
LAT = "cluster_lat"
LON = "cluster_lon"
CLUSTER_LABEL = "cluster_label"
STAY_DURATION = "stay_duration"
DAY_OF_WEEK = "sp_week_day"

ORIGINAL_LAT = "sp_lat"
ORIGINAL_LON = "sp_lon"

SCALE_ORDER = ["user_id",

               "trips_per_month",
               'ratio_stay_time_in_home',
               "trip_length",
               "rg_time",
               "rg_quantity",
               'shopping',
               'recreation',
               'restaurant',

               'ratio_of_uninterested_trips',
               'rg4/rg_quantity',
               'k_quantity',
               'random_entropy',
               'location_entropy',
               'OD_entropy',
               'sequence_entropy',
               'distance_from_home_entropy',

               "speed_std_mean",
               'speed_mean_std',
               "speed_std_max",
               "acceleration_std_max",
               "harsh_shift_ratio_std",
               "harsh_steering_ratio_std",
               "harsh_shift_ratio_mean",
               "harsh_steering_ratio_mean",

               "speed_mean",
               "speed_max",
               "over_speed_ratio",
               "over_speed_quantity",
               "junction_over_speed",
               "junction_speed_mean",
               "day_entropy",
               "datetime_entropy"]

# 图表中的名称修改的话，只需要修改这里

E1 = "Trips per month"
E2 = "Ratio of stay time at home"
E3 = "Trip length"
E4 = "Time-weighted radius of gyration"
E5 = "Radius of gyration"
E6 = "Average importance of shopping POI"
E7 = "Average importance of recreation POI"
E8 = "Average importance of restaurant POI"

O1 = "Ratio of uninterested trips"
O2 = "Ratio of 4-radius of gyration"
O3 = "Minimum number of extent-dominating locations"
O4 = "Random entropy"
O5 = "Location entropy"
O6 = "OD entropy"
O7 = "Sequence entropy"
O8 = "Distance from home entropy"

N1 = "Average standard deviation of speed"
N2 = "standard deviation of Average speed"
N3 = "Maximum standard deviation of speed"
N4 = "Maximum standard deviation of acceleration"
N5 = "Standard deviation ratio of harsh shift points"
N6 = "Standard deviation ratio of harsh steering points"
N7 = "Average ratio of harsh shift points"
N8 = "Average ratio of harsh steering points"

C1 = "Average speed"
C2 = "Average of maximum speed"
C3 = "Average ratio of speeding points"
C4 = "Average number of speeding points"
C5 = "Intersection's average ratio of speeding points"
C6 = "Intersection's average speed"
C7 = "Time-of-day entropy"
C8 = "Date-time entropy"

# EXTROVERSION = "Extroversion"
# OPENNESS = "Openness"
# NEUROTICISM = "Neuroticism"
# CONSCIENTIOUSNESS = "Conscientiousness"
#
# EXTROVERSION_LABEL = "Extroversion label"
# OPENNESS_LABEL = "Openness label"
# NEUROTICISM_LABEL = "Neuroticism label"
# CONSCIENTIOUSNESS_LABEL = "Conscientiousness label"

EXTROVERSION = "extroversion"
OPENNESS = "openness"
NEUROTICISM = "neuroticism"
CONSCIENTIOUSNESS = "conscientiousness"

EXTROVERSION_LABEL = "extroversion label"
OPENNESS_LABEL = "openness label"
NEUROTICISM_LABEL = "neuroticism label"
CONSCIENTIOUSNESS_LABEL = "conscientiousness label"

TRAIT_NAMES = {
    "user_id": "Driver",

    'trips_per_month': E1,
    'ratio_stay_time_in_home': E2,
    'trip_length': E3,
    "rg_time": E4,
    "rg_quantity": E5,
    'shopping': E6,
    'recreation': E7,
    'restaurant': E8,


    'ratio_of_uninterested_trips': O1,
    'rg4/rg_quantity': O2,
    "k_quantity": O3,
    "random_entropy": O4,
    'location_entropy': O5,
    "OD_entropy": O6,
    'sequence_entropy': O7,
    "distance_from_home_entropy": O8,


    "speed_std_mean": N1,
    'speed_mean_std': N2,
    "speed_std_max": N3,
    "acceleration_std_max": N4,
    "harsh_shift_ratio_std": N5,
    "harsh_steering_ratio_std": N6,
    "harsh_shift_ratio_mean": N7,
    "harsh_steering_ratio_mean": N8,

    "speed_mean": C1,
    "speed_max": C2,
    "over_speed_ratio": C3,
    "over_speed_quantity": C4,
    "junction_over_speed": C5,
    "junction_speed_mean": C6,
    "day_entropy": C7,
    "datetime_entropy": C8,

    "extroversion": EXTROVERSION,
    "openness": OPENNESS,
    "neuroticism": NEUROTICISM,
    "conscientiousness": CONSCIENTIOUSNESS,

    "extroversion_label": EXTROVERSION_LABEL,
    "openness_label": OPENNESS_LABEL,
    "neuroticism_label": NEUROTICISM_LABEL,
    "conscientiousness_label": CONSCIENTIOUSNESS_LABEL
}
