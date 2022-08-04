import { useState, useEffect } from 'react';

/**
 * 将原始数据组织为：[[lng,lat,key], ...] 的形式，并以密集散点图的形式展示
 * @param {*} chart - EChart实例
 * @param {*} data - 原始数据
 * @param {*} key - 特征键名
 * @param {*} seriesName - 图层名
 * @returns 
 */
export function useFeatureLayer(chart, orgData, key, seriesName) {
  const [feature, setFeature] = useState([]);
  useEffect(() => {
    if (orgData) {
      const coords = orgData.data;
      const spd = orgData[key];
      let data = [];
      for (let i = 0; i < coords.length; i++) {
        data.push([...coords[i], spd[i]])
      }
      setFeature(data);
    }
  }, [orgData])
  // 速度热力图层(密集散点实现)
  useEffect(() => {
    if (chart && feature.length) {
      chart.setOption({
        series: [{
          name: seriesName,
          data: feature,
        }]
      })
    }
  }, [chart, feature])

  return feature;
}