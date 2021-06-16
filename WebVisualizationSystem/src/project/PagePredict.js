// 第三方库
import React, { useRef, useEffect, useState, useContext, useReducer } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// 组件
import { Drawer, Select, message } from 'antd';
import SingleTrajSelector from '@/components/pagePredict/SingleTrajSelector'
import TimeSelector from '@/components/pagePredict/TimeSelector';
import TransferSelector from '@/components/pagePredict/TransferSelector';
import BrushBar from '@/components/bmapBrush/BrushBar';
import ModelCard from '@/components/pagePredict/ModelCard';
// 自定义
import { getOrg, getDest, getTraj } from '@/network';
import { useReqData } from '@/common/hooks/useReqData';
import { useStaticTraj } from '@/common/hooks/useStaticTraj';
import { useExceptFirst } from '@/common/hooks/useExceptFirst';
import { useTime } from '@/common/hooks/useTime';
import { globalStaticTraj, globalDynamicTraj } from '@/common/options/globalTraj';
import { trajColorBar } from '@/common/options/trajColorBar';
// Context 对象导入
import { drawerVisibility } from '@/context/mainContext'
// 样式
import './bmap.scss';




// 地图实例
let bmap = null;
// 目的地预测动效 - 计时器
let predictTimer = null;

export default function PagePredict(props) {
  // echarts 实例对象
  const [chart, setChart] = useState(null);
  // Context 对象
  const drawerVisibleObj = useContext(drawerVisibility);


  // scatter style: org / dest color
  const orgColor = '#00FFFF'; // 起点颜色
  const destColor = '#FF0033'; // 终点颜色
  const curColor = '#E53935'; // 当前点颜色

  /**
   * @param {function} requestMethod - 数据请求方法
   * org / dest
   * @returns {{id: number, coord: number[], count: number}[]} - id 数据唯一标识; coord 点坐标; count 默认1，用于热力图权重
   * traj
   * @returns {{id: number, data: number[][], distance: number, info: string}[]} - id 数据唯一标识; data 轨迹坐标集; distance 出行距离; info 轨迹描述信息(时间 / 距离 / ...)
   */
  const { data: org, isComplete: orgSuccess } = useReqData(getOrg);
  const { data: dest, isComplete: destSuccess } = useReqData(getDest);
  const { data: traj, isComplete: trajSuccess } = useReqData(getTraj, { min: 0, max: Infinity });


  // org / dest State: 起终点是否被选中
  const [brushData, setBrushData] = useState({});
  // org / dest selected res: {{org: number[], dest: number[]}}
  const [selected, setSelected] = useState({});

  // select by time
  const [byTime, setByTime] = useState([]);
  // select specific one deeply
  const [bySelect, setBySelect] = useState(-1); // 进一步筛选
  // select by brush
  const [byBrush, setByBrush] = useState({});
  // 存储用于预测的单条轨迹
  const [usedForPredict, setUsedForPredict] = useState({});
  // 存储当前轨迹坐标片段
  const [trajPart, setTrajPart] = useState({ idx: 0, coords: [] });


  // 模型面板的预测功能函数
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
    }
  }
  const [predict, predictDispatch] = useReducer(predictReducer, {
    startPredict: false,
    stopPredict: false,
    clearPredict: false,
  })


  // 容器 ref 对象
  const ref = useRef(null);

  // 静态配置项
  const option = {
    bmap: {
      center: [120.13066322374, 30.240018034923],
      zoom: 12,
      minZoom: 0,
      maxZoom: 20,
      roam: true, // 若设为 false，则地图底图不可移动或缩放
      mapStyle: {},
    },
    // legend
    legend: [{
      // 轨迹图例
      // 图例相对容器距离
      left: 'auto',
      right: '20rem',
      top: 'auto',
      bottom: '5rem',
      // 图例布局方向
      orient: 'vertical',
      // 文本样式
      textStyle: {
        color: '#fff',
      },
      data: [],
    }, {
      // OD 图例
      // 图例相对容器距离
      left: 'auto',
      right: '105rem',
      top: 'auto',
      bottom: '5rem',
      // 图例布局方向
      orient: 'vertical',
      // 文本样式
      textStyle: {
        color: '#fff',
      },
      data: [],
    }, {
      // OD 热力图图例
      // 图例相对容器距离
      left: 'auto',
      right: '170rem',
      top: 'auto',
      bottom: '5rem',
      // 图例布局方向
      orient: 'vertical',
      // 文本样式
      textStyle: {
        color: '#fff',
      },
      data: [],
    }],
    animation: false,
    visualMap: [
      // OD Cluster Heatmap
      {
        // https://echarts.apache.org/zh/option.html#visualMap
        type: 'continuous',
        // 视觉映射定义域
        min: 0,
        max: 10,
        // 不显示 visualMap 组件
        show: true,
        left: 20,
        bottom: 10,
        // 映射维度
        dimension: 2,
        seriesIndex: [2, 3], // OD聚类热力图
        // 定义域颜色范围
        inRange: {
          color: ['#00FFFF', '#33CC99', '#FFFF99', '#FF0033'],
        },
        textStyle: {
          color: "#fff",
        }
      },
    ],
    series: [{
      // 0. org
      name: '起点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: orgColor,
      }
    }, {
      // 1. dest
      name: '终点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: destColor,
      }
    }, {
      // 2. org-heatmap
      name: 'O聚类热力图',
      // https://echarts.apache.org/zh/option.html#series-heatmap
      type: 'heatmap',
      coordinateSystem: 'bmap',
      pointSize: 10,
      blurSize: 10,
      // 高亮状态图形样式
      emphasis: {
        // 高亮效果
        focus: 'series',
      },
      data: [],
    }, {
      // 3. dest-heatmap
      name: 'D聚类热力图',
      // https://echarts.apache.org/zh/option.html#series-heatmap
      type: 'heatmap',
      coordinateSystem: 'bmap',
      pointSize: 10,
      blurSize: 10,
      // 高亮状态图形样式
      emphasis: {
        // 高亮效果
        focus: 'series',
      },
      data: [],
    }, {
      // 4. selected org
      name: '筛选起点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: orgColor,
      }
    }, {
      // 5. selected dest
      name: '筛选终点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: destColor,
      }
    }, {
      // 6. select by time
      name: '轨迹时间筛选',
      type: "lines",
      coordinateSystem: "bmap",
      polyline: true,
      data: [],
      silent: true,
      lineStyle: {
        color: '#D4AC0D',
        opacity: 0.2,
        width: 1,
      },
      progressiveThreshold: 200,
      progressive: 200,
    }, {
      // 7. paint single static traj
      name: '静态单轨迹',
      type: 'lines',
      coordinateSystem: 'bmap',
      polyline: true,
      data: [],
      silent: true,
      lineStyle: {
        color: '#E0F7FA',
        opacity: 0.8,
        width: 3,
      },
      zlevel: 998,
    }, {
      // 8. paint single dynamic traj
      name: '动态单轨迹',
      type: "lines",
      coordinateSystem: "bmap",
      polyline: true,
      data: [],
      lineStyle: {
        width: 0,
        color: '#FB8C00'
      },
      effect: {
        constantSpeed: 100,
        // period: 1,
        show: true,
        trailLength: 0.8,
        symbolSize: 5,
      },
      zlevel: 999,
    }, {
      // 9. paint single dynamic scatter - origin point
      name: '出发地',
      type: 'effectScatter',
      // 何时显示动效：render - 绘制完成后，emphasis - 高亮显示
      showEffectOn: 'render',
      rippleEffect: {
        // 动效周期
        period: 4,
        // 波纹缩放比例
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      // 文本标签
      label: {
        show: true,
        position: 'top',
        distance: 5,
        formatter: '{a}',
        color: '#fff',
        offset: [20, -10],
      },
      // 标签视觉引导线
      labelLine: {
        show: true,
        showAbove: true,
        smooth: .1,
        length2: 20,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1000,
    }, {
      // 10. paint single dynamic scatter - destnation point
      name: '目的地',
      type: 'effectScatter',
      // 何时显示动效：render - 绘制完成后，emphasis - 高亮显示
      showEffectOn: 'render',
      rippleEffect: {
        // 动效周期
        period: 4,
        // 波纹缩放比例
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      // 文本标签
      label: {
        show: true,
        position: 'top',
        distance: 5,
        formatter: '{a}',
        color: '#fff',
        offset: [20, -10],
      },
      // 标签视觉引导线
      labelLine: {
        show: true,
        showAbove: true,
        smooth: .1,
        length2: 20,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1000,
    }, {
      // 11. paint select static traj
      name: '筛选轨迹',
      type: 'lines',
      coordinateSystem: 'bmap',
      polyline: true,
      data: [],
      silent: true,
      lineStyle: {
        color: '#FFF59D',
        opacity: 0.8,
        width: 1.5,
      },
      zlevel: 110
    }, {
      // 12. paint single dynamic scatter - current point
      name: '当前点',
      type: 'effectScatter',
      // 何时显示动效：render - 绘制完成后，emphasis - 高亮显示
      showEffectOn: 'render',
      rippleEffect: {
        // 动效周期
        period: 4,
        // 波纹缩放比例
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1000,
    },
    // global static trajectories
    ...globalStaticTraj,
    ...globalDynamicTraj,
    ]
  }


  /**
   * 单轨迹动效绘制
   * @param {number[][]} res - 轨迹经纬度坐标 [[lng, lat], ...]
   */
  function singleTraj(res) {
    chart.setOption({
      series: [{
        name: '静态单轨迹',
        data: [{
          coords: res,
        }],
      }, {
        name: '动态单轨迹',
        data: [{
          coords: res,
        }],
      },]
    })
  }
  /**
   * 单轨迹始发点与目的地绘制
   * @param {number[][]} res - 轨迹经纬度坐标 [[lng, lat], ...]
   */
  function singleOD(res) {
    chart.setOption({
      series: [{
        name: '出发地',
        data: [{
          value: res[0],
          itemStyle: {
            color: '#F5F5F5'
          }
        }],
      }, {
        name: '目的地',
        data: [{
          value: res.slice(-1)[0],
          itemStyle: {
            color: '#00CC33'
          }
        }],
      }]
    })
  }
  /**
   * 单轨迹当前点绘制
   * @param {number[][]} res - 轨迹经纬度坐标 [[lng, lat], ...]
   */
  function singleCurpt(res) {
    // 绘制当前轨迹末尾坐标
    chart.setOption({
      series: [{
        name: '当前点',
        data: [{
          value: res.slice(-1)[0],
          itemStyle: {
            color: curColor,
          }
        }],
      }]
    })
  }

  // 清除单轨迹动效绘制
  function clearSingleTraj() {
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
      }]
    })
  }
  // 依据 byBrush 筛选结果进行单轨迹动效绘制
  function singleTrajByBrush(val) {
    if (!val) return;
    const res = byBrush[val]?.data;
    // 记录轨迹
    setUsedForPredict(byBrush[val] || {});
    singleTraj(res)
    singleOD(res);
    // 取消全局图例选择
    glbUnSelectLegend(chart);
  }

  // 轨迹切分
  function trajSeparation(data, partnum) {
    if (data) {
      const partlens = Math.ceil(data.length / partnum);
      let start = partlens;
      let res = []
      while (start <= data.length + partlens) {
        res.push(data.slice(0, start));
        start += partlens
      }
      return res;
    } else {
      return [];
    }
  }

  useEffect(() => {
    // console.log(predict);
    // 生成切分轨迹段
    const separationRes = trajSeparation(usedForPredict?.data, 30);

    if (separationRes.length === 0 && !chart) return () => { }
    // 开始预测
    // !predictTimer 避免重复点击
    if (predict.startPredict && !predictTimer) {
      // 加载 Loading 模拟数据预测
      chart.showLoading();

      singleOD(usedForPredict?.data);
      predictTimer = setInterval(() => {
        setTrajPart(({ idx, coords }) => {

          if (coords.length !== 0) {
            // 取消 Loading
            chart.hideLoading();

            // 必须采用零延时，否则多个 setOption 会冲突
            setTimeout(() => {
              singleTraj(coords);
              singleCurpt(coords);
            }, 0)
          }

          return idx === separationRes.length ? {
            idx: 1,
            coords: separationRes[0],
          } : {
            idx: idx + 1,
            coords: separationRes[idx],
          }
        })
      }, 2000)
    }
    // 暂停预测
    if (predict.stopPredict) {
      clearInterval(predictTimer);
      predictTimer = null;
    }
    // 清除预测
    if (predict.clearPredict) {
      clearInterval(predictTimer);
      predictTimer = null;
      clearSingleTraj();
      singleTraj(usedForPredict.data);
      singleOD(usedForPredict.data)
      setTrajPart({
        idx: 0,
        coords: [],
      })
    }
  }, [predict, usedForPredict, chart])


  useEffect(() => {
    // 实例化 chart
    setChart(echarts.init(ref.current));
  }, [])

  useEffect(() => {
    if (!chart) return () => { }
    chart.setOption(option);
    // 加载 Loading
    chart.showLoading();

    // 获取地图实例, 初始化
    bmap = chart.getModel().getComponent('bmap').getBMap();
    bmap.centerAndZoom(props.initCenter, props.initZoom);
    bmap.setMapStyleV2({
      styleId: 'f65bcb0423e47fe3d00cd5e77e943e84'
    });
  }, [chart])

  // 当所有数据请求完毕后，再取消 Loading 动画展示
  useEffect(() => {
    orgSuccess && destSuccess && trajSuccess && chart.hideLoading();
  }, [orgSuccess, destSuccess, trajSuccess])

  // 全局静态轨迹
  const t0 = useStaticTraj(chart, { ...trajColorBar[0] });
  // const t1 = useStaticTraj(chart, { ...trajColorBar[1] });
  // const t2 = useStaticTraj(chart, { ...trajColorBar[2] });
  // const t3 = useStaticTraj(chart, { ...trajColorBar[3] });
  // const t4 = useStaticTraj(chart, { ...trajColorBar[4] });
  // const t5 = useStaticTraj(chart, { ...trajColorBar[5] });

  // paint Origin Points
  useEffect(() => {
    if (!orgSuccess) return () => { };
    const data = org.map(item => item.coord)
    chart.setOption({
      series: [{
        name: '起点',
        data,
      }]
    })
  }, [org, orgSuccess])
  // paint Destination Points
  useEffect(() => {
    if (!destSuccess) return () => { };
    const data = dest.map(item => item.coord)
    chart.setOption({
      series: [{
        name: '终点',
        data,
      }]
    })
  }, [dest, destSuccess])
  // paint Origin Heatmap
  useEffect(() => {
    if (!orgSuccess) return () => { };
    const data = org.map(item => [...item.coord, item.count])
    chart.setOption({
      series: [{
        name: 'O聚类热力图',
        data,
      }]
    })
  }, [org, orgSuccess])
  // paint Destination Heatmap
  useEffect(() => {
    if (!destSuccess) return () => { };
    const data = dest.map(item => [...item.coord, item.count])
    chart.setOption({
      series: [{
        name: 'D聚类热力图',
        data,
      }]
    })
  }, [dest, destSuccess])
  // 根据筛选结果更新样式
  useEffect(() => {
    console.log(selected);
    // 解构出被选中的 OD 点
    const { org: sorg = [], dest: sdest = [] } = selected;
    // 此处逻辑判断是否加载完毕原始数据
    const arr = [org, dest];
    const hasData = arr.reduce((prev, cur) => {
      let res = !prev ? false : prev && (cur.length !== 0)
      return res;
    }, true)
    // 只有当有 O（或 D）被选中，且存在原始数据时才发生渲染
    if (hasData && (sorg.length || sdest.length)) {
      // 筛选得到的起点
      const selectedOrg = org.filter(
        item => [...new Set([...sorg, ...sdest])].includes(item.id)
      ).map(item => item.coord);
      // 筛选得到的终点
      const selectedDest = dest.filter(
        item => [...new Set([...sorg, ...sdest])].includes(item.id)
      ).map(item => item.coord);
      // 筛选得到的轨迹
      const selectedTraj = traj.filter(
        item => [...new Set([...sorg, ...sdest])].includes(item.id)
      ).map(item => item.data);

      // 渲染筛选的数据
      chart.setOption({
        series: [{
          // org
          name: '起点',
          itemStyle: {
            color: '#fff',
          }
        }, {
          // dest
          name: '终点',
          itemStyle: {
            color: '#fff',
          }
        }, {
          // selected org
          name: '筛选起点',
          data: selectedOrg
        }, {
          // selected dest
          name: '筛选终点',
          data: selectedDest
        }, {
          // selected dest
          name: '筛选轨迹',
          data: selectedTraj
        }]
      })

      // 将筛选轨迹的结果存储
      const target = traj
        .filter(
          item => [...new Set([...sorg, ...sdest])].includes(item.id)
        )
        .reduce((prev, cur) => {
          prev[cur.info] = cur;
          return prev
        }, {})
      setByBrush(target)
    }
  }, [selected, org, dest, traj])
  // 取消框选时恢复样式
  function onClear() {
    chart && chart.setOption({
      series: [{
        // org
        name: '起点',
        itemStyle: {
          color: orgColor,
        }
      }, {
        // dest
        name: '终点',
        itemStyle: {
          color: destColor,
        }
      }, {
        // selected org
        name: '筛选起点',
        data: []
      }, {
        // selected dest
        name: '筛选终点',
        data: []
      }, {
        // selected dest
        name: '筛选轨迹',
        data: []
      }]
    })
  }


  function mapToObj(strMap) {
    let obj = {};
    for (let [k,v] of strMap) {
      obj[k] = v;
    }
    return obj;
  }
  // 图例
  useEffect(() => {
    if (!chart) return () => { }
    // 按距离划分的轨迹图例
    const data = trajColorBar.map(item => ({
      name: item.static,
      itemStyle: {
        color: item.color,
      }
    }))
    const selectedMap = new Map(trajColorBar.map(item => ([item.static.toString(), true])));
    const selected = mapToObj(selectedMap);
    chart.setOption({
      legend: [{
        data,
        selected,
      }, {
        data: [{
          name: '起点'
        }, {
          name: '终点'
        }],
        selected: {
          '起点': true,
          '终点': true,
        }
      }, {
        data: [{
          name: 'O聚类热力图'
        }, {
          name: 'D聚类热力图'
        }],
        selected: {
          'O聚类热力图': true,
          'D聚类热力图': true,
        }
      }]
    })
  }, [chart])


  // 全局图例 name
  const glbLegends = [...(trajColorBar.map(item => item.static)), '起点', '终点', 'O聚类热力图', 'D聚类热力图']
  // 取消图例选择
  function unSelectLegend(chart, name) {
    chart?.dispatchAction({
      type: 'legendUnSelect',
      name,
    })
  }
  // 图例选择
  function selectLegend(chart, name) {
    chart?.dispatchAction({
      type: 'legendSelect',
      name,
    })
  }
  // 全局图例取消选择
  function glbUnSelectLegend(chart) {
    glbLegends.forEach(item => unSelectLegend(chart, item));
  }
  // 全局图例选择
  function glbSelectLegend(chart) {
    glbLegends.forEach(item => selectLegend(chart, item));
  }




  // 存储时间信息
  /**timer:
   * dateStart
   * dateEnd
   * hourStart
   * hourEnd
   */
  const [timer, timerDispatch] = useTime();
  // 时间筛选静态轨迹 & 绘制
  useEffect(() => {
    if (!chart) return () => { }
    chart.setOption({
      series: [{
        name: '轨迹时间筛选',
        data: (bySelect === -1) ? byTime.map(item => item.data) : [],
      }]
    })
  }, [chart, byTime, bySelect])

  // 选择单条轨迹并绘制
  // bySelect - 选择轨迹对应的索引
  useExceptFirst(([bySelect, byTime, chart]) => {
    let res = byTime.find(item => item.id === bySelect);
    // 记录轨迹
    setUsedForPredict(res || {});

    res = res ? res.data : undefined;
    // res === undefined 表示没有找到数据
    if (!res) {
      clearSingleTraj();
    } else {
      // 绘制轨迹
      singleTraj(res)
      // 绘制 OD
      singleOD(res)
      // 取消全局图例选择
      glbUnSelectLegend(chart);
    }
  }, bySelect, byTime, chart)


  // 根据图例向 brush 组件传入选中的数据
  useEffect(() => {
    function legendselectchanged(e) {
      let obj = {
        '起点': (state) => {
          console.log(state);
          if (state) {
            setBrushData(prev => ({ ...prev, org }));
          } else {
            setBrushData(prev => {
              Reflect.deleteProperty(prev, 'org');
              return prev
            });
          }
        },
        '终点': (state) => {
          if (state) {
            setBrushData(prev => ({ ...prev, dest }));
          } else {
            setBrushData(prev => {
              Reflect.deleteProperty(prev, 'dest')
              return prev;
            });
          }
        },
      }
      // console.log(e);
      obj?.[e.name]?.(e.selected[e.name])
    }

    setBrushData(prev => {
      const arr = [org, dest, prev]
      const res = arr.filter(item => {
        const obj = {
          'array': () => (item.length !== 0),
          'object': () => (Object.keys(item).length !== 0),
        }

        let type = Object.prototype.toString.call(item).toString().slice(8, -1).toLowerCase()
        // console.log(type);
        return obj[type]();
      })

      return res.length ? { org, dest } : prev
    })

    // 添加事件
    // 可选链对象判空的应用
    chart?.on('legendselectchanged', legendselectchanged);

    return () => {
      chart?.off('legendselectchanged', legendselectchanged);
    }
  }, [chart, org, dest])


  return (
    <>
      <div
        key={'3-1'}
        ref={ref}
        className='bmap-container'
      ></div>
      {/* Brush 框选功能栏 */}
      <BrushBar
        map={bmap}
        // data={{org, dest}}
        data={brushData}
        getSelected={(value) => { setSelected(value) }}
        onClear={onClear}
      />
      {/* 左侧 Drawer */}
      <Drawer
        // Drawer 标题
        title="功能栏"
        // 弹出位置
        placement="left"
        // 是否显示右上角的关闭按钮
        closable={true}
        // 点击遮罩层或右上角叉或取消按钮的回调
        onClose={() => drawerVisibleObj.setLeftDrawerVisible(false)}
        // Drawer 是否可见
        visible={drawerVisibleObj.leftDrawerVisible}
        // 指定 Drawer 挂载的 HTML 节点, false 为挂载在当前 dom
        getContainer={ref.current}
        // 宽度
        width={'16rem'}
        // 是否显示遮罩层
        mask={false}
        // 点击蒙版是否允许关闭
        maskClosable={false}
        style={{
          position: 'absolute',
        }}
        // 多层 Drawer 嵌套的推动行为，默认会水平推动180px
        push={false}
      >
        <TimeSelector
          setByTime={setByTime}
          timer={timer}
          timerDispatch={timerDispatch}
        />
        {/* <TransferSelector
          data={byTime}
          setData={setBySelect}
        /> */}
        <SingleTrajSelector
          data={byTime}
          onSelect={(val) => setBySelect(val)}
          onClear={() => {
            setBySelect(-1); // 清空选择项
            setUsedForPredict({}); // 清空存储
          }}
        />
        <Select
          style={{ width: '100%' }}
          onChange={(val) => singleTrajByBrush(val)}
          onClear={clearSingleTraj}
          allowClear
        >
          {Object.entries(byBrush).map(item => (
            <Select.Option
              key={item[1].id}
              value={item[1].info}
            >{item[1].info}</Select.Option>
          ))}
        </Select>
        {/* 模型选择面板 */}
        <ModelCard
          width='100%'
          imgUrl='http://placehold.jp/150x150.png'
          // meta options
          avatarUrl='https://avatars.githubusercontent.com/u/42670632?v=4'
          title='Model Name'
          description='Model Description'
          startPredict={() => {
            if (Object.keys(usedForPredict).length === 0) {
              message.error('请选择有效的数据格式！', 1);
            } else {
              predictDispatch({ type: 'startPredict' });
            }
          }}
          stopPredict={() => {
            if (Object.keys(usedForPredict).length === 0) {
              message.error('请选择有效的数据格式！', 1);
            } else {
              predictDispatch({ type: 'stopPredict' });
            }
          }}
          clearPredict={() => {
            if (Object.keys(usedForPredict).length === 0) {
              message.error('请选择有效的数据格式！', 1);
            } else {
              predictDispatch({ type: 'clearPredict' });
            }
          }}
        ></ModelCard>
      </Drawer>
      {/* 右侧 Drawer */}
      <Drawer
        // Drawer 标题
        title="功能栏"
        // 弹出位置
        placement="right"
        // 是否显示右上角的关闭按钮
        closable={true}
        // 点击遮罩层或右上角叉或取消按钮的回调
        onClose={() => drawerVisibleObj.setRightDrawerVisible(false)}
        // Drawer 是否可见
        visible={drawerVisibleObj.rightDrawerVisible}
        // 指定 Drawer 挂载的 HTML 节点, false 为挂载在当前 dom
        getContainer={ref.current}
        // 宽度
        width={'15rem'}
        // 是否显示遮罩层
        mask={false}
        // 点击蒙版是否允许关闭
        maskClosable={false}
        style={{
          position: 'absolute',
        }}
      >
        <p>put something...</p>
      </Drawer>
    </>
  )
}
