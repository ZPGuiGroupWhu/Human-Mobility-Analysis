import { useState, useEffect } from 'react';

/**
 * data request
 * @param {function} reqMethod - 请求数据方法
 */
export const useReqData = (reqMethod, params=null) => {
  const [data, setData] = useState([]);
  const [isComplete, setComplete] = useState(false);
  // 请求数据
  useEffect(() => {
    async function getData() {
      console.time();
      // 数据请求
      let data = await reqMethod(params);
      setData(data);
      setComplete(true);
      console.timeEnd();
    }
    getData();
  }, [])
  return {
    data,
    isComplete,
  }
}

