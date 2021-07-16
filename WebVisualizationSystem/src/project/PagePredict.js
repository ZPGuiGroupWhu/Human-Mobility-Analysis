// 第三方库
import React, { useRef, useEffect, useState, useContext, useReducer } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// 组件
import { Drawer, Select, message, Switch, Slider, Space, Input, Checkbox } from 'antd';
// 通用函数
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom';
import { eventEmitter } from '@/common/func/EventEmitter';
import SearchPOI from '@/components/pagePredict/poi-selector';
// 通用 Hooks
import { useReqData } from '@/common/hooks/useReqData';
import { useExceptFirst } from '@/common/hooks/useExceptFirst';
import { useTime } from '@/common/hooks/useTime';
import { usePoi } from '@/components/pagePredict/hooks/usePoi';
// 通用配置
import { globalStaticTraj, globalDynamicTraj } from '@/common/options/globalTraj';
import { trajColorBar } from '@/common/options/trajColorBar';
// 网络请求
import { getOrg, getDest, getTraj, selectByTime } from '@/network';
// 自定义组件
import SingleTrajSelector from '@/components/pagePredict/SingleTrajSelector'
import TimeSelector from '@/components/pagePredict/TimeSelector';
import BrushBar from '@/components/bmapBrush/BrushBar'; // 框选功能条
import ModelCard from '@/components/pagePredict/ModelCard';
import MyDrawer from '@/components/drawer/MyDrawer';
import CardRightSelectedTraj from '@/components/pagePredict/CardRightSelectedTraj'; // 轨迹筛选结果展示卡片
import Calendar from '@/components/pagePredict/Calendar'; // 日历组件
import SimpleBar from '@/components/simpleBar/SimpleBar'; // 小型抽屉栏
import InfoBar from '@/components/infoBar/InfoBar'; // 框型小型抽屉栏
import ModelBar from '@/components/pagePredict/ModelBar'; // 模型功能条
import LegendBar from '@/components/pagePredict/LegendBar'; // 图例功能条
import TrajBar from '@/components/pagePredict/TrajBar'; // 轨迹功能条
// Context 对象导入
import { drawerVisibility } from '@/context/mainContext';
// 样式
import './bmap.scss';




// 地图实例
let bmap = null;
// SearchPOI 类实例
let searchPOI = null;
// 目的地预测动效 - 计时器
let predictTimer = null;
// iconfont Symbol 在线 url
const iconScriptUrl = '//at.alicdn.com/t/font_2577661_dmweq4qmkar.js';

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
   * traj
   * @returns {{id: number, data: number[][], distance: number, info: string}[]} - id 数据唯一标识; data 轨迹坐标集; distance 出行距离; info 轨迹描述信息(时间 / 距离 / ...)
   */
  // const { data: traj, isComplete: trajSuccess } = useReqData(getTraj, { min: 0, max: Infinity });

  // ------------------- 自定义 Hooks ---------------------
  const { poiDisabled, setPoiDisabled, poiState, poiDispatch } = usePoi();

  // ------------------- useReducer ---------------------
  // 1.
  function controller(state, action) {
    const { type, payload } = action;
    switch (type) {
      case 'legend':
        return {
          ...state,
          legend: !state.legend
        }
      case 'spatial':
        return {
          ...state,
          spatial: !state.spatial
        }
    }
  }
  const initCtrlState = {
    legend: false, // 图例面板开关
    spatial: false, // 空间查询开关
  }
  const [ctrlState, ctrlDispatch] = useReducer(controller, initCtrlState);



  // --------------------- useState ---------------------
  /**
   * org / dest
   * @returns {{id: number, coord: number[], count: number}[]} - id 数据唯一标识; coord 点坐标; count 默认1，用于热力图权重
   */
  // (所有)起点数据
  const [org, setOrg] = useState([]);
  // (所有)终点数据
  const [dest, setDest] = useState([]);
  // (所有)轨迹数据：用于框选
  const [traj, setTraj] = useState([]);

  // 获取当前年份
  const [curYear, setCurYear] = useState(null);
  // org / dest State: 起终点是否被选中
  const [brushData, setBrushData] = useState({});
  // org / dest selected res: {{org: number[], dest: number[]}}
  const [selected, setSelected] = useState({});
  // select by time
  const [byTime, setByTime] = useState([]);
  // select by brush
  const [byBrush, setByBrush] = useState([]);
  // select specific one deeply
  const [bySelect, setBySelect] = useState(-1); // 进一步筛选
  // 存储用于预测的单条轨迹
  const [usedForPredict, setUsedForPredict] = useState({});
  // 存储当前轨迹坐标片段
  const [trajPart, setTrajPart] = useState({ idx: 0, coords: [] });
  // 存储历史预测点集合，用于历史预测路径展示
  const [histPreres, setHistPreres] = useState([]);


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
          startPredict: false,
          clearPredict: true
        }
    }
  }
  const [predict, predictDispatch] = useReducer(predictReducer, {
    startPredict: false,
    stopPredict: false,
    clearPredict: false,
  })


  // POI查询-多选框-选项
  const checkBoxOptions = [
    { label: '起点', value: 'start' },
    { label: '当前点', value: 'current' },
    { label: '终点', value: 'end' },
  ]

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
    legend: [
      {
        // OD 图例
        // 图例相对容器距离
        left: '10px',
        right: 'auto',
        top: 'auto',
        bottom: '60px',
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
        left: '10px',
        right: 'auto',
        top: 'auto',
        bottom: '10px',
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
        show: false,
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
        // color: '#D4AC0D',
        color: '#FDFEFE',
        opacity: .4,
        width: 1,
        cap: 'round',
        join: 'round',
      },
      // 高亮样式
      emphasis: {
        focus: 'series',
        blurScope: 'series',
        lineStyle: {
          opacity: 1,
        },
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
        cap: 'round',
        join: 'round',
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
        color: '#FB8C00',
        cap: 'round',
        join: 'round',
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
        opacity: 0.6,
        width: 1.5,
        cap: 'round',
        join: 'round',
      },
      // 高亮样式
      emphasis: {
        lineStyle: {
          color: '#FF0000',
          width: 2,
          opacity: 1,
        },
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
      symbolSize: 5,
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1000,
    }, {
      // 13. 历史预测点集合
      name: '历史预测点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 8,
      itemStyle: {
        color: destColor,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1001,
    }, {
      // 14. 历史预测点集合的路径
      name: '历史预测轨迹',
      type: 'lines',
      coordinateSystem: 'bmap',
      polyline: true,
      data: [],
      silent: true,
      lineStyle: {
        color: '#FFF59D',
        opacity: 0.6,
        width: 1.5,
        cap: 'round',
        join: 'round',
      },
      // 高亮样式
      emphasis: {
        lineStyle: {
          color: '#FF0000',
          width: 2,
          opacity: 1,
        },
      },
      zlevel: 1002
    }, {
      // 15. 当前预测点
      name: '当前预测点',
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
      itemStyle: {
        color: destColor,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 1003,
    },
    ]
  }


  // --------------------- 自定义函数 ---------------------
  // 获取 bmap 实例
  function getBMapInstance(chart = null) {
    try {
      if (!chart) throw new Error('echarts实例不存在');
      // 获取地图实例, 初始化
      let bmap = chart.getModel().getComponent('bmap').getBMap();
      bmap.setMapStyleV2({
        styleId: 'f65bcb0423e47fe3d00cd5e77e943e84'
      });
      return bmap;
    } catch (err) {
      console.log(err);
    }
  }

  // 请求 OD 数据
  async function getOD(reqMethod, params = null) {
    let data;
    try {
      // 数据请求
      data = await reqMethod(params);
    } catch (err) {
      console.log(err);
    }
    return (data || [])
  }

  async function saveOD() {
    // 若未缓存数据，则请求数据
    if (!org.length) {
      const data = await getOD(getOrg);
      setOrg(data);
    }
    if (!dest.length) {
      const data = await getOD(getDest);
      setDest(data);
    }
  }

  // 请求所有轨迹
  function getalltraj(curYear) {
    return function (setFunc) {
      if (!curYear) return;
      const dayTime = 3600 * 60 * 1000
      let start = curYear + '-01-01'
      let end = +echarts.number.parseDate((+curYear + 1) + '-01-01') - dayTime;
      end = echarts.format.formatTime('yyyy-MM-dd', end);
      selectByTime(start, end).then(
        res => {
          // 将接收到的数据更新到 PagePredict 页面 state 中管理
          // setByTime(res || [])
          setFunc(res || []);
        }
      ).catch(
        err => console.log(err)
      );
    }
  }
  // 制定
  const getAllTraj = getalltraj(curYear);

  // 清除渲染项
  function clearOption(chart) {
    return function (name) {
      chart?.setOption({
        series: [{
          name,
          data: [],
        }]
      })
    }
  }

  // 颜色条 - 依据距离筛选
  function colorSelectByDistance(dis) {
    try {
      switch (true) {
        case dis < 5:
          return '#FDFD00';
        case dis < 10:
          return '#9DFD00';
        case dis < 20:
          return '#00FDDB';
        case dis < 30:
          return '#8600FD';
        case dis < 40:
          return '#FD008A';
        case dis < Infinity:
          return '#F8F9F9';
        default:
          throw new Error('something wrong')
      }
    } catch (err) {
      console.log(err);
    }
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
      }, {
        name: '历史预测点',
        data: [],
      }, {
        name: '历史预测路径',
        data: [],
      }, {
        name: '当前预测点',
        data: [],
      }]
    })
  }
  // 依据 byBrush 筛选结果进行单轨迹动效绘制
  function singleTrajByBrush(idx) {
    const res = byBrush[idx].data;
    // 记录轨迹
    setUsedForPredict(byBrush[idx] || {});
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


  /**
   * 历史预测路径展示
   * @param {number[][]} res - 轨迹经纬度坐标 [[lng, lat], ...]
   */
  function histPredictTraj(res) {
    chart.setOption({
      series: [
        {
          name: '历史预测点',
          data: res,
        },
        // {
        //   name: '历史预测轨迹',
        //   data: res,
        // },
        {
          name: '当前预测点',
          data: [{
            value: res.slice(-1)[0],
            itemStyle: {
              color: '#F5F5F5'
            }
          }]
        }
      ]
    })
  }

  // 模拟预测点坐标（后期用真实数据替换）
  function getMockData(val, min, max) {
    if (Array.isArray(val)) {
      return val.map((item, idx) => {
        return getMockData(item, min, max)
      })
    } else {
      return val + Math.random() * (max - min) + min
    }
  }

  // --------------------- Hooks ---------------------
  // 预测三阶段：开始 / 暂停 / 清除
  useEffect(() => {
    // 生成切分轨迹段
    // --- 后续轨迹切分段可动态设置
    const separationRes = trajSeparation(usedForPredict?.data, 30);
    if (separationRes.length === 0 && !chart) return () => { }

    // 开始预测
    // !predictTimer 避免重复点击
    if (predict.startPredict && !predictTimer) {
      singleOD(usedForPredict?.data); // 绘制预测轨迹的 OD 点


      // trajPart 用于记录每次历史动效，方便在暂停后恢复
      if (trajPart.idx < 1) {
        // id 为 0 表明本次为第一次展示
        setTimeout(() => {
          singleTraj(separationRes[0]);
          singleCurpt(separationRes[0]);
        }, 0)

        setTrajPart(({ idx }) => {
          return {
            idx: idx + 1,
            coords: separationRes[0],
          }
        })

        setHistPreres((prev) => {
          let mock = getMockData(usedForPredict?.data.slice(-1)[0], 0, 1)
          let data = [...prev, mock];
          setTimeout(() => {
            histPredictTraj(data);
          }, 50);
          return data;
        });
      }

      let i = 0; // 用于生成模拟数据

      predictTimer = setInterval(() => {

        setTrajPart(({ idx, coords }) => {
          setTimeout(() => {
            singleTraj(separationRes[idx]);
            singleCurpt(separationRes[idx]);
          }, 0)

          return idx === separationRes.length - 1 ? {
            idx: 0,
            coords: [],
          } : {
            idx: idx + 1,
            coords: separationRes[idx],
          }
        })

        setHistPreres((prev) => {
          if (i === separationRes.length - 1) i = 0;

          let mock = getMockData(separationRes[++i].slice(-1)[0], 0, 0.05);
          let data = [...prev, mock];
          setTimeout(() => {
            histPredictTraj(data);
          }, 50);
          return (i === separationRes.length - 1) ? [] : data;
        });
      }, 1000)
    }

    // 暂停预测
    if (predict.stopPredict) {
      clearInterval(predictTimer);
      predictTimer = null;
    }

    // 清除预测
    if (predict.clearPredict && trajPart.idx !== 0) {
      clearInterval(predictTimer);
      predictTimer = null;
      clearSingleTraj();
      singleTraj(usedForPredict.data);
      singleOD(usedForPredict.data)
      setTrajPart({
        idx: 0,
        coords: [],
      })

      setHistPreres([])
    }
  }, [trajPart, predict, usedForPredict, chart])

  // 首次进入页面，创建 echarts 实例
  useEffect(() => {
    // 实例化 chart
    setChart(() => {
      const chart = echarts.init(ref.current);
      chart.setOption(option);
      bmap = getBMapInstance(chart);
      bmap.centerAndZoom(props.initCenter, props.initZoom);
      return chart;
    });
  }, [])

  // OD 显示
  const [odShow, setodShow] = useState(false);
  useEffect(() => {
    if (!org.length || !dest.length) return () => { };

    function showOD(chart, { org, dest, odShow }) {
      if (!odShow) {
        chart.setOption({
          series: [{
            name: '起点',
            data: [],
          }, {
            name: '终点',
            data: [],
          }]
        })
      } else {
        const orgData = org.map(item => item.coord);
        const destData = dest.map(item => item.coord);
        setCenterAndZoom(bmap, [...orgData, ...destData]);
        chart.setOption({
          series: [{
            name: '起点',
            data: orgData,
          }, {
            name: '终点',
            data: destData,
          }]
        })
      }
    }

    showOD(chart, { org, dest, odShow })
  }, [org, dest, odShow])

  // OD-heatmap 显示
  const [heatmapShow, setHeatmapShow] = useState(false);
  useEffect(() => {
    if (!org.length || !dest.length) return () => { };
    if (!heatmapShow) {
      chart.setOption({
        series: [{
          name: 'O聚类热力图',
          data: [],
        }, {
          name: 'D聚类热力图',
          data: [],
        }]
      })
    } else {
      const orgData = org.map(item => [...item.coord, item.count])
      const destData = org.map(item => [...item.coord, item.count])
      setCenterAndZoom(bmap, [...orgData, ...destData]);
      chart.setOption({
        series: [{
          name: 'O聚类热力图',
          data: orgData,
        }, {
          name: 'D聚类热力图',
          data: destData,
        }]
      })
    }
  }, [org, dest, heatmapShow])


  // 图例
  useEffect(() => {
    if (!chart) return () => { }
    if (ctrlState.legend) {
      chart.setOption({
        legend: [
          {
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
    } else {
      chart.setOption({
        legend: [
          {
            data: []
          }, {
            data: [],
          }]
      })
    }

  }, [chart, ctrlState])


  // 根据筛选结果更新样式
  useEffect(() => {
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
      const filterData = traj.filter(
        item => [...new Set([...sorg, ...sdest])].includes(item.id)
      )
      const selectedTraj = filterData.map(item => item.data);

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
      // console.log(filterData);
      setByBrush(filterData)
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
  // 轨迹粗细 & 轨迹数量 联动
  const getTrajWidth = (count) => {
    const min = .5;
    const max = 4;
    const width = (-0.002) * count + 3.02
    switch (true) {
      case (width < min):
        return min;
      case (width > max):
        return max;
      default:
        return width;
    }
  };
  // 时间筛选静态轨迹 & 绘制
  useEffect(() => {
    if (!chart) return () => { };
    // 轨迹数据
    let data = byTime.map(item => ({
      coords: item.data,
      lineStyle: {
        color: colorSelectByDistance(item.distance)
      }
    }));
    // 数据为请求成功时
    if (!data.length) return () => { };

    console.log(data, bySelect);

    setCenterAndZoom(bmap, byTime.map(item => item.data).flat(1));
    chart.setOption({
      series: [{
        name: '轨迹时间筛选',
        data: (bySelect === -1) ? data : [],
        lineStyle: {
          width: getTrajWidth(data.length),
        }
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
      setCenterAndZoom(bmap, res)
      // 绘制轨迹
      singleTraj(res)
      // 绘制 OD
      singleOD(res)
      // 取消全局图例选择
      glbUnSelectLegend(chart);
    }
  }, bySelect, byTime, chart)


  // 单条轨迹 + POI 查询
  useEffect(() => {
    // 只有单条轨迹时才触发
    if (bySelect !== -1) {
      let res = byTime.find(item => item.id === bySelect);
      res = res ? res.data : undefined;
      // 是否启用 POI 查询
      if (poiDisabled) {
        try {
          poiState.description.forEach((item) => {
            let center;
            switch (item) {
              case 'start':
                center = res[0];
                break;
              case 'current':
                center = res[0];
                break;
              case 'end':
                center = res.slice(-1)[0];
                break;
              default:
                throw new Error('没有对应的类型')
            }
            searchPOI?.addAndSearchInCircle({
              keyword: poiState.keyword,
              center,
              radius: poiState.radius,
            })
          })

        } catch (err) {
          console.log(err);
        }
      }
    }
    return () => {
      searchPOI?.removeOverlay();
    }
  }, [searchPOI, bySelect, byTime, poiDisabled, poiState])


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

  useEffect(() => {
    if (Object.keys(byBrush).length === 0) return () => { }
    eventEmitter.emit('showTrajSelectByTime');
  }, [byBrush])


  // 实例化 SearchPOI 类
  useEffect(() => {
    if (!bmap) return () => { };
    searchPOI = new SearchPOI(bmap);
  }, [bmap])

  return (
    <>
      <div
        key={'3-1'}
        ref={ref}
        className='bmap-container'
      ></div>
      {/* 左侧 Drawer */}
      <Drawer
        // 弹出位置
        placement="left"
        // 是否显示右上角的关闭按钮
        closable={false}
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
        // 弹出位置
        placement="right"
        // 是否显示右上角的关闭按钮
        closable={false}
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
      {/* 日历热力图容器 */}
      <MyDrawer
        mode='bottom'
        modeStyle={{
          boxHeight: '200px',
          left: '180px',
          right: '2rem',
          btnOpenForbidden: true,
        }}
      >
        {/* 日历热力图 */}
        <Calendar
          byTime={byTime}
          callback={{
            setByTime,
            setCurYear,
            setBySelect
          }}
        />
      </MyDrawer>

      {/* 左侧功能栏 */}
      <MyDrawer
        mode='left'
        modeStyle={{
          boxWidth: '70px',
          top: '5px',
          bottom: '200px',
          backgroundColor: 'rgba(255, 255, 255, 0)',
        }}
      >

        {/* 图例面板 */}
        <SimpleBar
          width={80}
          iconScriptUrl={iconScriptUrl}
          iconType='icon-tuli'
          title='图例面板'
          callback={() => {
            ctrlDispatch({ type: 'legend' }) // 联动显示图例
            ctrlDispatch({ type: 'spatial' }) // 联动展开空间查询功能栏
          }}
        >
          <LegendBar
            fnList={[
              () => {
                saveOD();
                getAllTraj(setTraj);
                setodShow(prev => !prev);
              },
              () => {
                saveOD();
                setHeatmapShow(prev => !prev);
              }
            ]}
          />
        </SimpleBar>

        {/* 空间查询 */}
        <SimpleBar
          width={80}
          iconScriptUrl={iconScriptUrl}
          iconType='icon-kongjianchaxun'
          title='空间查询'
        >
          {/* Brush 框选功能栏 */}
          <BrushBar
            map={bmap}
            // data={{org, dest}}
            data={brushData}
            getSelected={(value) => { setSelected(value) }}
            onClear={onClear}
          />
        </SimpleBar>

        {/* 日历面板 */}
        <SimpleBar
          width={80}
          iconScriptUrl={iconScriptUrl}
          iconType='icon-rili'
          title='日历面板'
          callback={() => {
            eventEmitter.emit('showCalendar');
          }}
        >
          <TrajBar
            showAllTraj={() => { getAllTraj(setByTime) }}
            clearTraj={() => {
              clearOption(chart)('轨迹时间筛选');
              setByTime([]);
            }}
          />
        </SimpleBar>

        {/* 模型面板 */}
        <SimpleBar
          width={120}
          iconScriptUrl={iconScriptUrl}
          iconType='icon-moxingguanli'
          title='模型面板'
        >
          {/* 模型面板功能栏 */}
          <ModelBar
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
          />
        </SimpleBar>
        <InfoBar
          width={280}
          height={180}
          iconScriptUrl={iconScriptUrl}
          iconType='icon-pointofinterest'
          title='POI查询'
        >
          <Space
            align='center'
            direction='horizontal'
            size='small'
          >
            <span>{`POI查询(${poiDisabled ? '开' : '关'})`}</span>
            <Switch
              size="small"
              checked={poiDisabled}
              onChange={setPoiDisabled}
            />
          </Space>
          <Space
            align='center'
            direction='horizontal'
            size={1}
          >
            <div style={{width: '100px'}}>{`半径(${poiState.radius} 米)`}</div>
            <Slider
              min={1}
              max={500}
              defaultValue={poiState.radius}
              disabled={!poiDisabled}
              onChange={(value) => { poiDispatch({ type: 'radius', payload: value }) }}
              tooltipPlacement='left'
              tooltipVisible={false}
              style={{ width: '80px' }}
            />
          </Space>
          <Space
            align='center'
            direction='horizontal'
            size='small'
          >
            <span>{`关键词`}</span>
            <Input
              placeholder='POI关键词'
              disabled={!poiDisabled}
              onChange={(e) => { poiDispatch({ type: 'keyword', payload: e.target.value }) }}
              allowClear
              size='middle'
              style={{ width: '100px' }}
            />
          </Space>
          <Checkbox.Group
            options={checkBoxOptions}
            defaultValue={poiState.description}
            disabled={!poiDisabled}
            onChange={(val) => { poiDispatch({ type: 'description', payload: val }); }}
          />
        </InfoBar>
      </MyDrawer>
      {/* 展示轨迹筛选结果的卡片 */}
      <CardRightSelectedTraj
        chart={chart}
        byTime={byTime}
        byBrush={byBrush}
        setBySelect={setBySelect}
        singleTrajByBrush={singleTrajByBrush}
      />
    </>
  )
}
