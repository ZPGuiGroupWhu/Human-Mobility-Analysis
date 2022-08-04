import { useReducer, useEffect, useState, useRef } from 'react';

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
   * @param {number[]} pt 当前预测点数据 [id, [lng, lat], error]
   */
  function drawPredict(pt) {
    chart.setOption({
      series: [{
        name: '当前预测点',
        data: [{
          value: [...pt[1], pt[2]],
          itemStyle: { color: '#FF0000' },
        }]
      }]
    })
  }

  /**
   * 绘制历史预测点
   * @param {number[][]} pts 轨迹数组
   */
  function drawHistPredicts(pts) {
    const data = pts.map(item => {
      const val = 1 - ((pts.length - 1 - item[0]) * (1 / pts.length)).toFixed(1)
      return {
        value: [...item[1], item[2]],
        itemStyle: {
          color: `rgba(255,0,0,${val})`,
        },
      }
    })
    chart.setOption({
      series: [{
        name: '历史预测点',
        data,
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

  // 预测主要逻辑
  const idx = useRef(0); // 当前轨迹片段索引
  const [timer, setTimer] = useState(null); // 动画计时器
  useEffect(() => {
    if (!chart || !traj) return () => { }; // 不存在 ECharts 实例时执行
    // 开始预测
    if (predict.startPredict) {
      let t = setInterval(() => {
        const lens = trajParts.length;
        if (idx.current === lens) { idx.current = 0 }; // 循环绘制
        const curPredict = [idx.current, results[idx.current][2], results[idx.current][3]];
        const histPredicts = results.slice(0, idx.current).map((res, idx) => ([idx, res[2], res[3]])); // [编号，预测坐标，误差]
        // 每次循环绘制将其放入宏任务队列：因为在主线程中同时触发多个chart.setOption会报错 (移动地图也会重新触发Chart.setOption)
        setTimeout(() => {
          drawTraj(chart, trajParts[idx.current]); // 绘制轨迹
          drawCurpt(chart, trajParts[idx.current]); // 绘制当前点
          drawPredict(curPredict); // 绘制预测点
          drawHistPredicts(histPredicts); // 绘制历史预测点
          showDistanceError(traj.data.slice(-1)[0], results[idx.current][2]); // 误差可视化
          idx.current++;
        }, 0)
      }, 1000);
      setTimer(t);
    }

    // 暂停执行
    if (predict.stopPredict) {
      clearInterval(timer); // 清除计时器
    }

    // 结束执行
    if (predict.clearPredict) {
      clearSingleTraj(chart); // 清除绘制
      clearInterval(timer); // 清除计时器
      idx.current = 0; // 重置索引
      // 绘制原轨迹
      drawOD(chart, traj.data);
      drawTraj(chart, traj.data);
    }
  }, [traj, trajParts, predict.startPredict, predict.stopPredict, predict.clearPredict])

  return {
    predictDispatch,
  }
}