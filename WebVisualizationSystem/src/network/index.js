import { baseRequest, baseRequestWithCancel } from './base';

export const getUserTraj = (id) => {
  return baseRequest({
    url: `/getUserTraj?id=${id}`
  })
}

export const getUserTrajCount = (id) => {
  return baseRequest({
    url: `/getUserTrajCount?id=${id}`
  })
}

export const getUserTrajInChunk = (id,chunkSize,chunkNum) => {
  return baseRequest({
    url: `/getUserTrajInChunk?id=${id}&chunkNum=${chunkNum}&chunkSize=${chunkSize}`
  })
}

export const getUserODs = () => {
  return baseRequest({
    url: `/getUserODs`
  })
}

export const getOceanScoreAll = () => {
  return baseRequest({
    url: `/ocean_score_all`,
  })
}


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

export const getUserHistoryTraj = (id, days) => {
  return baseRequest({
    url: `/getUserHistoryTraj?days=${days}&trajId=${id}`
  })
}