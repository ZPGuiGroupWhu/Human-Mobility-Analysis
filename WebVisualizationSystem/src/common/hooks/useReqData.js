import { useState, useEffect } from 'react';

/**
 * data request
 * @param {function} reqMethod - 请求数据方法
 */
export const useReqData = (reqMethod, params = null) => {
  const [data, setData] = useState([]);
  const [isComplete, setComplete] = useState(false);
  // 请求数据
  useEffect(() => {
    async function getData() {
      try {
        // 数据请求
        let data = await reqMethod(params);
        // 若请求失败，数据为 undefined 时设置默认值
        setData(data || []);
        setComplete(true);
      } catch (err) {
        console.log(err);
      }
    }
    getData();
  }, [])
  return {
    data,
    isComplete,
  }
}

