import { baseRequest } from './base';

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