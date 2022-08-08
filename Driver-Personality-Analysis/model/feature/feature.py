# This is a Python script for extracting trajectory features.
import datetime
import pandas as pd
import numpy as np
from model.utils.gislib import getDistanceByHaversine
from model.utils.utils import time2seconds
from model.utils import constants
from scipy import stats
import geopandas as gpd


def read_L(file_name):
    """
    Read the stay points sequence file (.csv) for extract trajectory features.
    :param file_name: str
        The name with path of stay points sequence file.
    :return: DataFrame
        The DataFrame of L.
    """
    csv_data = pd.read_csv(file_name)
    csv_data[constants.CHECK_IN_TIME] = pd.to_datetime(csv_data[constants.CHECK_IN_TIME])
    csv_data[constants.CHECK_OUT_TIME] = pd.to_datetime(csv_data[constants.CHECK_OUT_TIME])
    csv_data[constants.STAY_DURATION] = pd.to_timedelta(csv_data[constants.STAY_DURATION])
    return csv_data


# temporal features
# temporal entropy
def day_of_week_entropy(L, normalize=False):
    """
    Compute the day-of-week entropy of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param normalize: boolean, optional
        If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N)`, where :math:`N` is the number of timeslots. The default is False.
    :return: float
        The day of week entropy of the individual.
    """
    probs = L[constants.DAY_OF_WEEK].value_counts(normalize=True).values
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = 7
        entropy /= np.log2(n_vals)
    return entropy


def _frequency_time_of_day(L, column=constants.CHECK_IN_TIME, period=3600):
    """
     Compute the frequency of travel in each timeslot on daily basis of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param column: str
        The column name in L for Compute the frequency of travel. The default is checkin time to stay point.
    :param period: int, optional
        The time slot interval (unit is seconds) for dividing timeslots at equal intervals. The default is 3600 seconds (1 hour).
    :return: list
        Each element in the list is a value of frequency corresponding the timeslot.
    """
    time_cut = pd.DataFrame()
    time_cut['time'] = L[column].apply(lambda x: time2seconds(x.time()))
    end_time = datetime.datetime.strptime('23:59:59', '%H:%M:%S')
    end_time_seconds = time2seconds(end_time)
    time_slot = list(range(0, end_time_seconds, period))
    time_slot.append(end_time_seconds)
    time_cut['time_cut'] = pd.cut(time_cut['time'], time_slot, labels=range(len(time_slot) - 1))
    time_cut['time_cut'] = time_cut['time_cut'].fillna(0)
    frequency = time_cut['time_cut'].value_counts(normalize=True).values
    return frequency


def time_of_day_entropy(L, normalize=False):
    """
        Compute the time-of-day entropy of a single individual given their L.
        :param L: DataFrame
            The DataFrame of stay points sequence.
        :param normalize: boolean, optional
            If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N)`, where :math:`N` is the number of timeslots. The default is False.
        :return: float
            The day of week entropy of the individual.
    """
    probs = _frequency_time_of_day(L)
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = 24
        entropy /= np.log2(n_vals)
    return entropy


def _frequency_datetime(L, column=constants.CHECK_IN_TIME, period=3600):
    """
         Compute the frequency of travel in each timeslot of a single individual given their L.
        :param L: DataFrame
            The DataFrame of stay points sequence.
        :param column: str, optional
            The column name in L for Compute the frequency of travel. The default is checkin time to stay point.
        :param period: int, optional
            The time slot  interval on a daily basis (unit is seconds) for dividing timeslots at equal intervals. The default is 3600 seconds (1 hour).
        :return: list
            Each element in the list is a value of frequency corresponding the timeslot.
    """
    datetime_cut = pd.DataFrame()
    datetime_cut['dayofweek'] = L[constants.DAY_OF_WEEK]
    datetime_cut['time'] = L[column].apply(lambda x: time2seconds(x.time()))
    end_time = datetime.datetime.strptime('23:59:59', '%H:%M:%S')
    end_time_seconds = time2seconds(end_time)
    time_slot = list(range(0, end_time_seconds, period))
    time_slot.append(end_time_seconds)
    datetime_cut['time_cut'] = pd.cut(datetime_cut['time'], time_slot, labels=range(len(time_slot) - 1))
    datetime_cut['time_cut'] = datetime_cut['time_cut'].fillna(0)
    result = datetime_cut.groupby(['dayofweek', 'time_cut'])['time'].aggregate('count')
    frequency = result.values
    n = frequency.sum()
    probs = frequency / n
    return probs


def datetime_entropy(L, normalize=False):
    """
        Compute the date-time entropy of a single individual given their L.
        :param L: DataFrame
            The DataFrame of stay points sequence.
        :param normalize: boolean, optional
            If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N)`, where :math:`N` is the number of timeslots. The default is False.
        :return: float
            The date-time entropy of the individual.
    """
    probs = _frequency_datetime(L)
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = 24 * 7
        entropy /= np.log2(n_vals)
    return entropy


# spatial features
#  radiuses of gyration
def radius_of_gyration(L, method='quantity'):
    """
    Compute the radius of gyration of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param method: str, optional
        This parameter controls whether the radius of rotation is time-weighted or quantity-weighted. The default is quantity.
    :return: float
        The radius of gyration of the individual.
    """
    if method == 'quantity':
        lats_lngs = L[[constants.LAT, constants.LON]].values
        center_of_mass = np.mean(lats_lngs, axis=0)
        rg = np.sqrt(np.mean([getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0 for lat, lng in lats_lngs]))
        return rg
    elif method == 'time':
        lats_lngs = L[[constants.LAT, constants.LON]].values
        stay_time = L[constants.STAY_DURATION].values
        stay_time_norm = (stay_time - stay_time.min()) / (stay_time.max() - stay_time.min())
        total_time = sum(stay_time_norm)
        center_of_mass = stay_time_norm.dot(lats_lngs) / total_time
        rg = np.sqrt(sum([stay_time_norm[i] * (getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0)
                          for i, (lat, lng) in enumerate(lats_lngs)]) / total_time)
        return rg


def k_radius_of_gyration(L, k=2, method='quantity'):
    """
        Compute the k-radius of gyration of a single individual given their L.
        :param L: DataFrame
            The DataFrame of stay points sequence.
        :param k: int, optional
            The number of most frequent locations to consider. The default is 2. The possible range of values is math:`[2, +inf]`.
         :param method: str, optional
            This parameter controls whether the radius of rotation is time-weighted or quantity-weighted. The default is quantity.
        :return: float
            The k-radius of gyration of the individual.
    """
    if method == 'quantity':
        L['visits'] = L.groupby([constants.LAT, constants.LON]).transform('count')[
            constants.CHECK_IN_TIME]
        top_k_locations = L.drop_duplicates(subset=[constants.LAT, constants.LON]).sort_values(
            by=['visits', constants.CHECK_IN_TIME],
            ascending=[False, True])[:k]
        visits = top_k_locations['visits'].values
        total_visits = sum(visits)
        lats_lngs = top_k_locations[[constants.LAT, constants.LON]].values

        center_of_mass = visits.dot(lats_lngs) / total_visits
        krg = np.sqrt(sum([visits[i] * (getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0)
                           for i, (lat, lng) in enumerate(lats_lngs)]) / total_visits)
        return krg
    elif method == 'time':
        stay_time = L[constants.STAY_DURATION].values
        stay_time_norm = (stay_time - stay_time.min()) / (stay_time.max() - stay_time.min())
        L['stay_time_norm'] = stay_time_norm
        L['visits'] = L.groupby([constants.LAT, constants.LON])['stay_time_norm'].transform('sum')
        top_k_locations = L.drop_duplicates(subset=[constants.LAT, constants.LON]).sort_values(
            by=['visits', constants.STAY_DURATION],
            ascending=[False, True])[:k]
        visits = top_k_locations['visits'].values
        total_visits = sum(visits)
        lats_lngs = top_k_locations[[constants.LAT, constants.LON]].values

        center_of_mass = visits.dot(lats_lngs) / total_visits
        krg = np.sqrt(sum([visits[i] * (getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0)
                           for i, (lat, lng) in enumerate(lats_lngs)]) / total_visits)
        return krg


def _n_radius_of_gyration(L):
    """
       Compute the k-radius of gyration with different k values of a single individual given their L. k increases from 2 to the total number of stay points.
       :param L: DataFrame
           The DataFrame of stay points sequence.
       :return: DataFrame
           The k-radius of gyration of the individual with different k values.
    """
    userid = L[constants.USER][0]
    # compute radius of gyration
    rg_time = radius_of_gyration(L, 'time')
    rg_quantity = radius_of_gyration(L, 'quantity')
    nrg = pd.DataFrame(columns=['user_id', 'k', 'kr_time', 'kr_quantity'])
    # the number of stay points
    n = len(L.groupby([constants.LAT, constants.LON]))
    if n > 2:
        for k in range(2, n):
            krg_quantity = k_radius_of_gyration(L, k)
            krg_time = k_radius_of_gyration(L, k, 'time')
            nrg.loc[len(nrg), nrg.columns] = userid, k, krg_time, krg_quantity
            # nrg = nrg.append({'k': k, 'kr_time': krg_time, 'kr_quantity': krg_quantity}, ignore_index=True)
    unique_sp = n
    nrg.loc[len(nrg), nrg.columns] = userid, unique_sp, rg_time, rg_quantity
    # nrg = nrg.append({'k': unique_sp, 'kr_time': rg_time, 'kr_quantity': rg_quantity}, ignore_index=True)
    nrg['user_id'] = userid
    return nrg


def K(L, method='quantity'):
    """
    Compute the minimum number of extent-dominating locations of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param method: str, optional
        This parameter controls whether the radius of rotation is time-weighted or quantity-weighted. The default is quantity.
    :return: int
        The minimum number of extent-dominating locations of the individual.
    """
    nrg = _n_radius_of_gyration(L)

    if len(nrg) < 2:
        return 1

    if method == 'quantity':
        rg_q = nrg.iloc[-1, 3]
        half_rg_quantity = rg_q / 2
        ks_quantity = nrg.loc[nrg['kr_quantity'] > half_rg_quantity, 'k']
        k_q = min(ks_quantity)
        return k_q
    elif method == 'time':
        rg_t = nrg.iloc[-1, 2]
        half_rg_time = rg_t / 2
        ks_time = nrg.loc[nrg['kr_time'] > half_rg_time, 'k']
        k_t = min(ks_time)
        return k_t


def ratio_of_k_radius_of_gyration(L, k=2, method='quantity'):
    """
     Compute the ratio of k-radius of gyration of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param k: int, optional
        The number of most frequent locations to consider. The default is 2. The possible range of values is math:`[2, +inf]`.
    :param method: str, optional
        This parameter controls whether the radius of rotation is time-weighted or quantity-weighted. The default is quantity.
    :return: float
        The ratio of k-radius of gyration of the individual.
    """
    nrg = _n_radius_of_gyration(L)
    if len(nrg) < k - 1:
        return 1
    if method == 'quantity':
        rg_q = nrg.iloc[-1, 3]
        rgk_rg_quantity_rat = nrg.iloc[k - 2, 3] / rg_q
        return rgk_rg_quantity_rat
    if method == 'time':
        rg_t = nrg.iloc[-1, 2]
        rgk_rg_time_rat = nrg.iloc[k - 2, 2] / rg_t
        return rgk_rg_time_rat


# spatial entropy
def random_entropy(L):
    """
    Compute the random entropy of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :return: float
        The random entropy of the individual.
    """
    n_distinct_locs = len(L.groupby([constants.LAT, constants.LON]))
    entropy = np.log2(n_distinct_locs)
    return entropy


def location_entropy(L, normalize=False):
    """
      Compute the location entropy of a single individual given L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param normalize: boolean, optional
        If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations. The default is False.
    :return: float
        The location entropy of the individual.
    """
    n = len(L)
    probs = [1.0 * len(group) / n for group in
             L.groupby(by=[constants.LAT, constants.LON]).groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(L[[constants.LAT, constants.LON]].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def _stringify(seq):
    return '|'.join(['_'.join(list(map(str, r))) for r in seq])


def _sequence_entropy(sequence):
    n = len(sequence)

    # these are the first and last elements
    sum_lambda = 1. + 2.

    for i in range(1, n - 1):
        str_seq = _stringify(sequence[:i])
        j = 1
        str_sub_seq = _stringify(sequence[i:i + j])
        while str_sub_seq in str_seq:
            j += 1
            str_sub_seq = _stringify(sequence[i:i + j])
            if i + j == n:
                # EOF character
                j += 1
                break
        sum_lambda += j

    return 1. / sum_lambda * n * np.log2(n)


def sequence_entropy(L):
    """
    Compute the sequence entropy of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :return: float
        The sequence entropy of the individual.
    """
    time_series = tuple(map(tuple, L[[constants.LAT, constants.LON]].values))
    entropy = _sequence_entropy(time_series)
    return entropy


def od_entropy(L, normalize=False):
    """
     Compute the OD entropy of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param normalize: boolean, optional
        If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct OD. The default is False.
    :return: float
        The OD entropy of the individual.
    """
    od = pd.DataFrame()
    od['origin_lat'] = L[constants.LAT]
    od['origin_lon'] = L[constants.LON]
    od['destination_lat'] = L[constants.LAT].shift(-1)
    od['destination_lon'] = L[constants.LON].shift(-1)
    od.dropna(inplace=True)
    n = len(od)
    probs = [1.0 * len(group) / n for group in
             od.groupby(by=['origin_lat', 'origin_lon', 'destination_lat',
                            'destination_lon']).groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(L[[constants.LAT, constants.LON]].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def _cut_data(data, column, start_date, date_num):
    open_day = start_date
    close_day = start_date + pd.Timedelta(days=date_num)
    con1 = data[column] >= open_day
    con2 = data[column] < close_day
    result = data[con1 & con2]
    return result


def sliding_window_entropy(L, column, size, length, func_entropy, *args, **kwargs):
    """
    Calculate entropy based on sliding window.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param column: the column name used for cutting L into each window.
    :param size: int
        The size of window,and the unit is day.
    :param length: int
        The sliding length,and the unit is day.
    :param func_entropy: str
        The function name of entropy to be calculated.
    :param args: optional
    :param kwargs: optional
    :return: float
        The average of entropies computed by multiple sliding windows.
    """
    t = datetime.time(0, 0, 0)
    start_date = datetime.datetime.combine(L[constants.CHECK_IN_TIME].get(0).date(), t)
    end_date = datetime.datetime.combine(L[constants.CHECK_IN_TIME][len(L) - 1].date() + pd.Timedelta(days=1), t)
    window_open = start_date
    slide_num = 0
    sliding_entropies = []
    while (window_open + pd.Timedelta(days=size)) <= end_date:
        sub_L = _cut_data(L, column, window_open, size)
        if len(sub_L) < 1:
            print('there is no data in this sliding window')
            window_open += pd.Timedelta(days=length)
            continue
        sub_entropy = func_entropy(sub_L, *args, **kwargs)
        sliding_entropies.append(sub_entropy)
        window_open += pd.Timedelta(days=length)
        slide_num += 1
    if slide_num < 1:
        print('Data is less than a sliding window')
        return None
    else:
        return np.mean(sliding_entropies)


# others
def trips_per_month(L):
    """
    Compute the trips per month of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :return: float
        The trips per month of the individual.
    """
    start_date = L[constants.CHECK_IN_TIME].get(0).date()
    end_date = L[constants.CHECK_IN_TIME][len(L) - 1].date()
    # interval_month = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
    interval_month = (end_date - start_date).days / 30
    return len(L) / interval_month


def trip_length(L):
    """
     Compute the trip length of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :return: float
        The trip length of the individual.
    """
    lat_lng = L[[constants.LAT, constants.LON]]
    N = len(lat_lng)
    total_distance = 0
    for i in range(N - 1):
        total_distance += getDistanceByHaversine(lat_lng.iloc[i, :], lat_lng.iloc[i + 1, :])
    return total_distance / (N - 1)


def ratio_of_uninterested_trips(L, min_count=3):
    """
    Compute the ratio of uninterested trips of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param min_count: int, optional
        This parameter is used for judging the uninterested trips of the individual.
    :return: float
        The ratio of uninterested trips of the individual
    """
    trip_count = L.groupby([constants.LAT, constants.LON])['checkin_time'].count().to_frame()
    uninterested_trip = trip_count.loc[trip_count['checkin_time'] < min_count, 'checkin_time'].sum()
    return uninterested_trip / len(L)


# semantic features
# distance from home
def _calc_overlap_times(start_night, end_night, start_time, end_time):
    today = start_time.date()
    if start_night > end_night:  # night spanning two days, e.g.（22：00-07：00）
        current_slot_start_date = (today + datetime.timedelta(days=-1)).strftime("%Y-%m-%d")
        current_slot_end_date = today.strftime('%Y-%m-%d')
        current_slot_start_night = datetime.datetime.strptime(current_slot_start_date + ' ' + start_night,
                                                              "%Y-%m-%d %H:%M")
        current_slot_end_night = datetime.datetime.strptime(current_slot_end_date + ' ' + end_night, "%Y-%m-%d %H:%M")

        next_slot_start_date = today.strftime('%Y-%m-%d')
        next_slot_end_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        next_slot_start_night = datetime.datetime.strptime(next_slot_start_date + ' ' + start_night, "%Y-%m-%d %H:%M")
        next_slot_end_night = datetime.datetime.strptime(next_slot_end_date + ' ' + end_night, "%Y-%m-%d %H:%M")

    else:  # night within one day, e.g.（01：00-07：00）
        current_slot_start_date = today.strftime('%Y-%m-%d')
        current_slot_end_date = today.strftime('%Y-%m-%d')
        current_slot_start_night = datetime.datetime.strptime(current_slot_start_date + ' ' + start_night,
                                                              "%Y-%m-%d %H:%M")
        current_slot_end_night = datetime.datetime.strptime(current_slot_end_date + ' ' + end_night, "%Y-%m-%d %H:%M")

        next_slot_start_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        next_slot_end_date = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        next_slot_start_night = datetime.datetime.strptime(next_slot_start_date + ' ' + start_night, "%Y-%m-%d %H:%M")
        next_slot_end_night = datetime.datetime.strptime(next_slot_end_date + ' ' + end_night, "%Y-%m-%d %H:%M")

    if start_time < current_slot_end_night:
        latest_start = max(current_slot_start_night, start_time)
        earliest_end = min(current_slot_end_night, end_time)
        overlap = earliest_end - latest_start
    else:
        latest_start = max(next_slot_start_night, start_time)
        earliest_end = min(next_slot_end_night, end_time)
        overlap = earliest_end - latest_start

    # In the absence of intersection, overlap equals zero
    if overlap < datetime.timedelta(days=0):
        overlap = datetime.timedelta(days=0)
    return overlap


def _home_location(L, start_night='01:00', end_night='06:00'):
    """
      Compute the home location of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param start_night: str, optional
        The starting time of the night (format HH:MM). The default is '01:00'.
    :param end_night: str, optional
        The ending time for the night (format HH:MM). The default is '06:00'.
    :return:  tuple
        The latitude and longitude coordinates of the individual's home location
    """
    L_copy = L.copy()
    overlap = L_copy.apply(
        lambda x: _calc_overlap_times(start_night, end_night, x[constants.CHECK_IN_TIME], x[constants.CHECK_OUT_TIME]),
        axis=1)
    L_copy['night_in_home'] = overlap
    night_in_home = L_copy.groupby([constants.LAT, constants.LON])['night_in_home'].sum().sort_values(
        ascending=False)
    time_in_home = night_in_home.iloc[0]
    if time_in_home > datetime.timedelta(days=0):
        home_coords = night_in_home.index[0]
    else:
        print(start_night + '-' + end_night + ': no stay during that time')
        lat, lng = L_copy.groupby([constants.LAT, constants.LON]).count().sort_values(by=constants.CHECK_IN_TIME,
                                                                                      ascending=False).iloc[0].name
        home_coords = (lat, lng)
    return home_coords


def max_distance_from_home(L, start_night='01:00', end_night='06:00'):
    """
    Compute the maximum distance from home traveled by a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param start_night: str, optional
        The starting time of the night (format HH:MM). The default is '22:00'.
    :param end_night: str, optional
        The ending time for the night (format HH:MM). The default is '07:00'.
    :return: float
        The maximum distance from home traveled by the individual.(km)
    """
    lats_lngs = L.sort_values(by=constants.CHECK_IN_TIME)[[constants.LAT, constants.LON]].values
    home = _home_location(L, start_night=start_night, end_night=end_night)
    home_lat, home_lng = home[0], home[1]
    lengths = np.array(
        [getDistanceByHaversine((lat, lng), (home_lat, home_lng)) for i, (lat, lng) in enumerate(lats_lngs)])
    return lengths.max()


def distance_from_home_entropy(L, start_night='01:00', end_night='06:00', normalize=False):
    """
     Compute the distance from home entropy of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param start_night: str, optional
        The starting time of the night (format HH:MM). The default is '22:00'.
    :param end_night: str, optional
        The ending time for the night (format HH:MM). The default is '07:00'.
    :param normalize: boolean, optional
        If True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct distance to home. The default is False.
    :return: float
        The distance from home entropy of the individual.
    """
    lats_lngs = L.sort_values(by=constants.CHECK_IN_TIME)[[constants.LAT, constants.LON]].values
    home = _home_location(L, start_night=start_night, end_night=end_night)
    home_lat, home_lng = home[0], home[1]
    lengths = pd.DataFrame(
        data=[getDistanceByHaversine((lat, lng), (home_lat, home_lng)) for i, (lat, lng) in enumerate(lats_lngs)],
        columns=['home_distance'])
    lengths = lengths.apply(round)
    n = len(lengths)
    probs = [1.0 * len(group) / n for group in
             lengths.groupby(by='home_distance').groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = int(lengths.nunique())
        print(n_vals)
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def ratio_stay_home_time(L, start_night='01:00', end_night='06:00', threshold=7):
    """
    Compute the ratio of stay time at home of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param start_night: str, optional
        The starting time of the night (format HH:MM). The default is '22:00'.
    :param end_night: str, optional
        The ending time for the night (format HH:MM). The default is '07:00'.
    :param threshold: int, optional
        To attenuate the effect of missing trajectories on the calculation of stay time, we only sum the stay time less than the threshold
    :return: float
        The ratio of stay time at home of the individual.
    """
    home = _home_location(L, start_night=start_night, end_night=end_night)
    home_lat, home_lng = home[0], home[1]

    sp_used = L[L[constants.STAY_DURATION] < pd.Timedelta(days=threshold)].copy()
    sp_in_home = sp_used[
        (sp_used[constants.LAT] == home_lat) & (sp_used[constants.LON] == home_lng)].copy()
    ratio_time_in_home = sp_in_home[constants.STAY_DURATION].sum() / sp_used[constants.STAY_DURATION].sum()
    return ratio_time_in_home


# POI feature
def _df2shp(L, lon, lat):
    shp_gdf = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L[lon],
                                                              L[lat],
                                                              crs="EPSG:4326"))
    return shp_gdf


def _extract_values_to_points(polygons, points):
    poly_pt_sjoin = gpd.sjoin(points, polygons, op='within', how='left')
    poly_pt_sjoin.drop(columns=['geometry', 'grid_id', 'index_right'], inplace=True)
    return poly_pt_sjoin


def POI_features(L, semantic_map):
    """
     Compute the average POI importance metrics of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :param semantic_map: GeoDataFrame
    :return: list
        The average POI importance metrics of the individual. including shopping, restaurant, recreation.
    """

    L_shp = _df2shp(L, constants.ORIGINAL_LON, constants.ORIGINAL_LAT)
    semantic_L = _extract_values_to_points(semantic_map, L_shp)
    POI_importance_list = ['shop', 'play', 'eat']
    POI_importance = semantic_L[POI_importance_list]
    average_POI_importance = []
    for column, val in POI_importance.iteritems():
        average_POI_importance.append(val.mean())
    return average_POI_importance


# driving behavior features
# overall behavior
# abnormal behavior
# intersection behavior
def driving_behavior_features(L):
    """
    Compute the driving behavior features of a single individual given their L.
    :param L: DataFrame
        The DataFrame of stay points sequence.
    :return: list
        The driving behavior features.
    """
    speed_mean_all_traj = L['speed_mean'].mean()
    speed_max_all_traj = L['speed_max'].mean()

    speed_std_all_traj = L['speed_std'].mean()
    speed_std_max_all_traj = L['speed_std'].max()
    speed_mean_std_all_traj = L['speed_mean'].std()

    acceleration_std_max_all_traj = L['acceleration_std'].max()

    L['harsh_shift_ratio'] = L['harsh_acceleration_ratio'] + L['harsh_breaking_ratio']
    hshift_all_traj = L['harsh_shift_ratio'].mean()
    hshift_std_all_traj = L['harsh_shift_ratio'].std()

    hs_all_traj = L['harsh_steering_ratio'].mean()
    hs_std_all_traj = L['harsh_steering_ratio'].std()

    os_all_traj = L['over_speed_ratio'].mean()
    os_all_traj_quantity = L['over_speed'].mean()

    junc_os = L['junction_over_speed'].mean()
    junc_sm = L['junction_speed_mean'].mean()

    return [speed_std_all_traj, speed_mean_std_all_traj,
            speed_std_max_all_traj, acceleration_std_max_all_traj,
            hshift_std_all_traj, hs_std_all_traj,
            hshift_all_traj, hs_all_traj,
            speed_mean_all_traj, speed_max_all_traj,
            os_all_traj, os_all_traj_quantity,
            junc_os, junc_sm]
