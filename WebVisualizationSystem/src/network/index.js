import { baseRequest, baseRequestWithCancel } from './base';

export const getUserTraj = (id) => {
  return baseRequest({
    url: `/getUserTraj?id=${id}`
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