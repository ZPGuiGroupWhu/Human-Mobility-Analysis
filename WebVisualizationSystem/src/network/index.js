import { baseRequest } from './base';

export const getOrgDemo = () => {
  return baseRequest({
    url: '/org'
  })
}

export const getDestDemo = () => {
  return baseRequest({
    url: '/dest'
  })
}

export const getTrajDemo = ({min, max}) => {
  return baseRequest({
    url: `/traj?min=${min}&&max=${max}`
  })
}