import axios from 'axios';

export function baseRequest(config) {
  const instance = axios.create({
    baseURL: 'http://192.168.61.60:8081/',
    // baseURL: 'http://10.244.135.231:8081/',
  })

  instance.interceptors.response.use(
    res => res.data,
    err => console.log(err)
  )

  return instance(config)
}


export function baseRequestWithCancel(config, id) {
  const instance = axios.create({
    baseURL: 'http://192.168.61.60:8081/',
    // baseURL: 'http://10.244.135.231:8081/',
  })

  instance.interceptors.request.use(
    config => {
      config.cancelToken = new axios.CancelToken(function executor(c) {
        // executor 函数接收一个 cancel 函数作为参数
        window.cancelList[id] = c
      });
      return config
    },
    err => console.log(err)
  )

  instance.interceptors.response.use(
    res => res.data,
    err => console.log(err)
  )

  return instance(config);
}