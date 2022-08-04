import { getPredictResult } from '@/network';

/**
 * 请求模型分段预测结果
 * @param {string} trajID - 轨迹编号，例如 '399313_2'
 * @param {number|string} size - 轨迹片段粒度，例如 '0.1' 表示轨迹按 1/10 划分
 */
function fetchData(trajID, size) {
  let promises = [];
  let lens = size;
  while (lens <= 1) {
    promises.push(getPredictResult(trajID, lens));
    lens += size ;
  }
  return Promise.all(promises);
}

export default fetchData;