import axios from 'axios';

export function baseRequest(config) {
  const instance = axios.create({
    baseURL: 'http://127.0.0.1:7001/',
    timeout: 50000,
  })

  instance.interceptors.response.use(
    res => res.data,
    err => console.log(err)
  )

  return instance(config)
}