import { baseRequest, baseRequestWithCancel } from './base';

// 获取指定用户编号的个体轨迹数据集
export const getUserTraj = (id) => {
  return baseRequest({
    url: `/getUserTraj?id=${id}`
  })
}

// 获取个体轨迹数据集的轨迹条数
export const getUserTrajCount = (id) => {
  return baseRequest({
    url: `/getUserTrajCount?id=${id}`
  })
}

// 分块获取个体轨迹数据
export const getUserTrajInChunk = (id,chunkSize,chunkNum) => {
  return baseRequest({
    url: `/getUserTrajInChunk?id=${id}&chunkNum=${chunkNum}&chunkSize=${chunkSize}`
  })
}

// 获取轨迹OD
export const getUserODs = () => {
  return baseRequest({
    url: `/getUserODs`
  })
}

// 获取个体大五人格信息
export const getOceanScoreAll = () => {
  return baseRequest({
    url: `/oceanScoreAll`,
  })
}

// 获取用户 top5 位置数据
export const getUsersTopFive = () => {
  return baseRequest({
    url: `/getUsersTopFive`,
  })
}

// 获取各天各用户出行次数
export const getUserTrajNumsByDay = () => {
  return baseRequest({
    url: `/getUserTrajNumsByDayFake`,
  })
}

// 获取个体轨迹集的时间统计特征
export const getUserTrajByTime = (params) => {
  if (window.cancelList.hasOwnProperty('getUserTrajByTime')) {
    window.cancelList.getUserTrajByTime()
    delete window.cancelList.getUserTrajByTime
  }
  return baseRequestWithCancel({
    url: '/getUserTrajByTime',
    params,
  }, 'getUserTrajByTime')
}

// 获取个体历史前N天的轨迹数据
export const getUserHistoryTraj = (id, days) => {
  return baseRequest({
    url: `/getUserHistoryTraj?days=${days}&trajId=${id}`
  })
}

// 获取指定轨迹编号的轨迹
export const getOneTraj = (trajId) => {
  return baseRequest({
    url: `/getOneTraj?trajId=${trajId}`
  })
}

// 轨迹编号模糊检索
export const getUserTrajRegex = (userid, num) => {
  return baseRequest({
    url: `/getUserTrajRegex?id=${userid}&searchNum=${num}`
  })
}

export const getUserTrajectoryCountBetweenDate = (id, startDate, endDate) => {
  return baseRequest({
    url: `/getUserTrajectoryCountBetweenDate?endDate=${endDate}&id=${id}&startDate=${startDate}`
  })
}

/**
 * 请求模型分段预测结果
 * @param {string} trajid - 轨迹编号，例如 '399313_2'
 * @param {number|string} curpt - 轨迹片段粒度，例如 '0.1' 表示请求轨迹 1/10 段时的预测结果
 * @returns 模型分段预测结果 [Promise]
 */
export const getPredictResult = (trajid, curpt) => {
  return baseRequest({
    method: 'get',
    url: `/getPredictResult?cutPoint=${curpt}&trajID=${trajid}`,
  });


}