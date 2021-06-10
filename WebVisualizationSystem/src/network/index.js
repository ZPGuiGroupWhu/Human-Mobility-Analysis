import { baseRequest } from './base';
import { isEmptyString } from '@/common/func/isEmptyString'

export const getOrg = () => {
  return baseRequest({
    url: '/org'
  })
}

export const getDest = () => {
  return baseRequest({
    url: '/dest'
  })
}

export const getTraj = ({ min, max }) => {
  return baseRequest({
    url: `/traj?min=${min}&&max=${max}`
  })
}

/**
 * 根据日期筛选轨迹
 * @param {string} dateStart - 日期起点
 * @param {string} dateEnd - 日期终点
 * @param {string} hourStart - 时起点
 * @param {string} hourEnd - 时终点
 * @returns 发起请求
 */
export const selectByTime = (dateStart, dateEnd, hourStart, hourEnd) => {
  let url = '/select?';
  switch (true) {
    case isEmptyString(dateStart, dateEnd, hourStart, hourEnd):
      url += `dateStart=${dateStart}&&dateEnd=${dateEnd}&&hourStart=${hourStart}&&hourEnd=${hourEnd}`;
      break;
    case isEmptyString(dateStart, dateEnd):
      url += `dateStart=${dateStart}&&dateEnd=${dateEnd}`;
      break;
    case isEmptyString(hourStart, hourEnd):
      url += `hourStart=${hourStart}&&hourEnd=${hourEnd}`;
      break;
    default:
      break;
  }
  console.log(url);
  return baseRequest({
    url,
  })
}