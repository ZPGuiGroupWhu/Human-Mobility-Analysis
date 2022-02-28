import { useReducer, useEffect, useState } from 'react';

export const useShowPredict = (chart, traj, nums, results, { drawOD, drawTraj, drawCurpt }) => {
  /**
   * 轨迹切分
   * @param {number[][]} data 轨迹数据
   * @param {number} partnum 分段数
   * @returns 切分结果
   */
  function trajSeparation(data, partnum) {
    const partlens = Math.ceil(data.length / partnum);
    let start = partlens;
    let res = []
    while (start <= data.length + partlens) {
      res.push(data.slice(0, start));
      start += partlens
    }
    return res;
  }

  // 清除单轨迹动效绘制
  function clearSingleTraj(chart) {
    chart.setOption({
      series: [{
        name: '静态单轨迹',
        data: [],
      }, {
        name: '动态单轨迹',
        data: [],
      }, {
        name: '出发地',
        data: [],
      }, {
        name: '目的地',
        data: [],
      }, {
        name: '当前点',
        data: [],
      }, {
        name: '历史预测点',
        data: [],
      }, {
        name: '当前预测点',
        data: [],
      }, {
        name: '距离可视化',
        data: [],
      }]
    })
  }

  /**
   * 绘制当前预测点
   * @param {number[]} pt 轨迹经纬度坐标 [lng, lat]
   */
  function drawPredict(pt) {
    chart.setOption({
      series: [{
        name: '当前预测点',
        data: [{
          value: pt,
          itemStyle: {
            color: '#F5F5F5'
          }
        }]
      }]
    })
  }

  /**
   * 绘制历史预测点
   * @param {number[][]} pts 轨迹经纬度坐标 [[lng, lat], ...]
   */
  function drawHistPredicts(pts) {
    chart.setOption({
      series: [{
        name: '历史预测点',
        data: pts,
      }]
    })
  }

  /**
   * 距离误差展示
   * @param {number[]} des 真实目的地坐标
   * @param {number[]} pre 预测点坐标
   */
  function showDistanceError(des, pre) {
    chart.setOption({
      series: [{
        name: '距离可视化',
        data: [{ coords: [pre, des] }],
      }]
    })
  }

  /**
   * 模型面板的预测功能函数
   * @param {any} state 
   * @param {{type:string, payload:any}} action
   * @returns 
   */
  function predictReducer(state, action) {
    const keys = Object.keys(state);
    keys.forEach(item => state[item] = false)
    switch (action.type) {
      case 'startPredict':
        return {
          ...state,
          startPredict: true
        }
      case 'stopPredict':
        return {
          ...state,
          stopPredict: true
        }
      case 'clearPredict':
        return {
          ...state,
          clearPredict: true
        }
      default:
        return;
    }
  }
  const initArgs = {
    startPredict: false,
    stopPredict: false,
    clearPredict: false,
  }
  const [predict, predictDispatch] = useReducer(predictReducer, initArgs);


  // 生成并存储切分的轨迹片段
  const [trajParts, setTrajParts] = useState([]); // 轨迹片段
  useEffect(() => {
    if (!traj || !nums) return () => { }
    setTrajParts(trajSeparation(traj.data, nums));
  }, [traj, nums])

  const [curPart, setCurPart] = useState(0); // 暂停的轨迹片段索引
  const [timer, setTimer] = useState(null); // 动画计时器
  useEffect(() => {
    if (!chart || !traj) return () => { }; // 不存在 ECharts 实例时执行
    let i = curPart; // 当前轨迹片段索引
    // 开始预测
    if (predict.startPredict) {
      let t = setInterval(() => {
        if (i === trajParts.length) { i = 0 }; // 循环绘制
        drawTraj(chart, trajParts[i]); // 绘制轨迹
        drawCurpt(chart, trajParts[i]); // 绘制当前点
        drawPredict(results[i][2]); // 绘制预测点
        drawHistPredicts(results.slice(0, i).map(res => (res[2]))); // 绘制历史预测点
        showDistanceError(traj.data.slice(-1)[0], results[i][2]); // 误差可视化
        i++;
      }, 1000);
      setTimer(t);
    }

    // 暂停执行
    if (predict.stopPredict) {
      clearInterval(timer); // 清除计时器
      setCurPart(i); // 记录当前暂停的轨迹片段索引
    }

    // 结束执行
    if (predict.clearPredict) {
      clearSingleTraj(chart); // 清除绘制
      clearInterval(timer); // 清除计时器
      setCurPart(0); // 重置索引
      // 绘制原轨迹
      drawOD(chart, traj.data);
      drawTraj(chart, traj.data);
    }
  }, [traj, trajParts, predict])

  return {
    predictDispatch,
  }
}