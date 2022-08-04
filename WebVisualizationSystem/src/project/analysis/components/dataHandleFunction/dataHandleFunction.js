import _ from 'lodash'

// 根据日期筛筛选轨迹编号
export function getSelectIdsByDate(data, start, end) {
    let selectTrajIds = [];
    let startTimeStamp = Date.parse(start);
    let endTimeStamp = Date.parse(end);
    for (let i = 0; i < data.length; i++) {
        if (startTimeStamp <= Date.parse(data[i].date) && Date.parse(data[i].date) <= endTimeStamp) {
            selectTrajIds.push(data[i].id);
        }
    }
    return selectTrajIds  //返回选择的轨迹编号
}

// calendarSelected: 初始化轨迹集
export function initData(data) {
    let selectedData = [];
    _.forEach(data, (item) => {
        selectedData.push(item.id)
    })
    return selectedData;
}

// characterSelected: 根据slider的month范围决定初始化的tripsLayer的轨迹集
export function getInitTrajIds(data, start, end) {
    let initTrajIds = [];
    _.forEach(data, (item) => {
      let month = parseInt(item.date.split('-')[1]);
      if (start <= month && month <= end) {
        initTrajIds.push(item.id)
      }
    })
    return initTrajIds;
}