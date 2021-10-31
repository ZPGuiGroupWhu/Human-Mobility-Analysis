import axios from 'axios';

export function baseRequest(config) {
  const instance = axios.create({
    baseURL: 'http://192.168.61.60:8081/',
  })

  instance.interceptors.response.use(
    res => res.data,
    err => console.log(err)
  )

  return instance(config)
}