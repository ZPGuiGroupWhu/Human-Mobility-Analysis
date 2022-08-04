// 第三方库
import React, { useRef, useEffect, useState, useReducer } from 'react';
import { useSelector } from 'react-redux';
import axios from 'axios';
import eventBus, { HISTACTION } from '@/app/eventBus';
// 通用函数
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom'; // 自主聚焦视野
import transcoords from '@/common/func/transcoords'; // 坐标纠偏
import { withMouse } from '@/components/drawer/withMouse'; // 高阶函数-监听鼠标位置
import fetchDataForPredict from './function/fetchData'; // 请求预测结果
// 逻辑分离
import { useCreate } from '@/project/predict/function/useCreate';
import { usePoiSearch } from '@/project/predict/function/usePoiSearch'; // poi 查询
import { useShowPredict } from '@/project/predict/function/useShowPredict'; // 轨迹预测
import { useFeatureLayer } from '@/project/predict/function/useFeatureLayer'; // 特征热力图层展示
import { useSingleTraj } from '@/project/predict/function/useSingleTraj'; // 从候选轨迹中选择一条轨迹
// 自定义组件
import Drawer from '@/components/drawer/Drawer'; // 抽屉
import Foobar from './components/foobar/Foobar'; // 左侧功能栏
import RelationChart from './components/charts/relation-chart/RelationChart'; // EChart关系折线图
import Doughnut from './components/charts/doughnut-chart/Doughnut'; // Echarts 环形统计图
import Tooltip from '@/components/tooltip/Tooltip'; // 自定义悬浮框
import ScatterTooltip from './components/scatter-tooltip/ScatterTooltip'; // 点-tooltip
import ShoppingDrawer from '../analysis/components/shopping/ShoppingDrawer';
// 样式
import '@/project/bmap.scss';


function PagePredict(props) {
  const [ShenZhen, setShenZhen] = useState(null); // 存放 ShenZhen.json 数据
  const [histTrajs, setHistTrajs] = useState([]); // 存放历史轨迹数据
  const { curShowTrajId } = useSelector(state => (state.analysis)) // 当前展示的轨迹编号
  useEffect(() => {
    axios.get(process.env.PUBLIC_URL + '/ShenZhen.json').then(data => setShenZhen(data.data)); // 请求ShenZhen.json
    // 获取前N天历史轨迹数据：数据组织+坐标纠偏
    eventBus.on(HISTACTION, (histTrajs) => {
      if (histTrajs.length) {
        let res = histTrajs.map((traj) => {
          const obj = {
            data: [],
            spd: traj.spd,
            azm: traj.azimuth,
            dis: traj.dis,
          };
          const lens = traj.spd.length;
          for (let i = 0; i < lens; i++) {
            obj.data.push([traj.lngs[i], traj.lats[i]]);
          }
          obj.data = transcoords(obj.data)
          return obj;
        })
        setHistTrajs(res);
      } else {
        setHistTrajs([]);
      }
    })
  }, [])


  const [drawerId, setDrawerId] = useState(1); // 当前展开的抽屉 id

  const selectedTraj = useSingleTraj(true); // 从候选列表中选取一条轨迹(用于展示)

  const ref = useRef(null); // 容器 ref 对象
  // 首次进入页面，创建 echarts 实例
  const initCenter = props.initCenter;
  const initZoom = props.initZoom;
  const { bmap, chart } = useCreate({ ref, initCenter, initZoom })


  // 保存props为不受监听的可变对象, 避免频繁触发依赖项
  const newProps = useRef(props);
  useEffect(() => {
    newProps.current = props;
  })
  // tooltip - reducer
  function tooltipReducer(state, action) {
    const { type, payload } = action;
    const params = payload ? {
      top: payload.top,
      left: payload.left,
      display: payload.display,
      data: payload.data || null,
    } : {};
    switch (type) {
      case 'showOrg':
        return {
          ...state,
          ...params,
          type: 'org', // 标记tooltip触发类型
        };
      case 'showDest':
        return {
          ...state,
          ...params,
          type: 'dest',
        }
      case 'showCur':
        return {
          ...state,
          ...params,
          type: 'cur',
        }
      case 'showCurPredict':
        return {
          ...state,
          ...params,
          type: 'curPredict',
        }
      case 'showHistPredicts':
        return {
          ...state,
          ...params,
          type: 'histPredicts',
        }
      case 'hidden':
        return {
          display: 'none',
        }
      default:
        return { ...state };
    }
  }
  const initTooltipArg = {
    type: '',
    top: '0px',
    left: '0px',
    display: 'none',
    data: null,
  }
  const [tooltip, tooltipDispatch] = useReducer(tooltipReducer, initTooltipArg);
  // 组件切换
  const tooltipOptions = {
    'org': () => (<ScatterTooltip title="出发地" lng={tooltip.data.value[0].toFixed(3)} lat={tooltip.data.value[1].toFixed(3)} />),
    'dest': () => (<ScatterTooltip title="目的地" lng={tooltip.data.value[0].toFixed(3)} lat={tooltip.data.value[1].toFixed(3)} />),
    'cur': () => (<ScatterTooltip title="当前点" lng={tooltip.data.value[0].toFixed(3)} lat={tooltip.data.value[1].toFixed(3)} />),
    'curPredict': () => (
      <ScatterTooltip title="当前预测点" lng={tooltip.data.value[0].toFixed(3)} lat={tooltip.data.value[1].toFixed(3)}>
        <div style={{ color: '#fff' }}><strong>预测误差:</strong> {tooltip.data.value[2].toFixed(3)}</div>
      </ScatterTooltip>
    ),
    'histPredicts': () => (
      <ScatterTooltip title="历史预测点" lng={tooltip.data.value[0].toFixed(3)} lat={tooltip.data.value[1].toFixed(3)}>
        <div style={{ color: '#fff' }}><strong>预测误差:</strong> {tooltip.data.value[2].toFixed(3)}</div>
      </ScatterTooltip>
    ),
  }
  const setExtraEChartsTooltip = (chart, seriesName, actionType, componentType) => {
    try {
      chart.on('click', { seriesName }, function (params) {
        tooltipDispatch({
          type: actionType,
          payload: {
            type: componentType,
            top: newProps.current.position.top + 'px',
            left: newProps.current.position.left + 'px',
            display: '',
            data: params.data,
          }
        })
        setTimeout(() => {
          chart.on('mouseout', { seriesName }, function fn(params) {
            tooltipDispatch({ type: 'hidden' });
            chart.off('mouseout', fn);
          });
        }, 0)
      });
    } catch (err) {
      console.log(err);
    }
  }
  // 添加 EChart 事件监听
  useEffect(() => {
    if (chart) {
      setExtraEChartsTooltip(chart, '出发地', 'showOrg', 'org');
      setExtraEChartsTooltip(chart, '目的地', 'showDest', 'dest');
      setExtraEChartsTooltip(chart, '当前点', 'showCur', 'cur');
      setExtraEChartsTooltip(chart, '当前预测点', 'showCurPredict', 'curPredict');
      setExtraEChartsTooltip(chart, '历史预测点', 'showHistPredicts', 'histPredicts');
    }
  }, [chart])


  /**
   * 单轨迹绘制
   * @param {number[][]} data - 轨迹经纬度坐标 [[lng, lat], ...]
   */
  // 绘制静态轨迹
  function drawTraj(chart, data) {
    chart.setOption({
      series: [{
        name: '静态单轨迹',
        data: [{
          coords: data,
        }],
      }, {
        name: '动态单轨迹',
        data: [{
          coords: data,
        }],
      },]
    })
  }
  // 绘制静态OD
  function drawOD(chart, data) {
    chart.setOption({
      series: [{
        name: '出发地',
        data: [{
          value: data[0],
          itemStyle: {
            color: '#F5F5F5'
          }
        }],
      }, {
        name: '目的地',
        data: [{
          value: data.slice(-1)[0],
          itemStyle: {
            color: '#00CC33'
          }
        }],
      }]
    })
  }
  // 绘制当前轨迹末尾坐标
  function drawCurpt(chart, res) {
    chart.setOption({
      series: [{
        name: '当前点',
        data: [{
          value: res.slice(-1)[0],
          itemStyle: {
            color: '#E53935',
          }
        }],
      }]
    })
  }

  // 绘制当前选中轨迹
  useEffect(() => {
    if (chart && selectedTraj) {
      const { data } = selectedTraj;
      setCenterAndZoom(bmap, data); // 调整视野
      drawOD(chart, data)
      drawTraj(chart, data);
    }
  }, [chart, selectedTraj])


  // 前N天历史轨迹展示
  useEffect(() => {
    if (chart) {
      chart.setOption({
        series: [{
          name: '前N天历史静态多轨迹',
          data: histTrajs.map(traj => ({
            coords: traj.data,
          })),
        }]
      })
    }
  }, [chart, histTrajs])

  // ---------------------------------------------
  // 轨迹预测: 开始 / 暂停 / 清除
  const cuts = 0.1; // 粒度
  const nums = Math.floor(1 / cuts); // 分段数
  const [predicts, setPredicts] = useState([]); // 预测结果
  // 获取预测结果
  useEffect(() => {
    async function fetchData() {
      let data = await fetchDataForPredict(curShowTrajId, cuts);
      data = data.map((item, idx) => {
        let { des, pred, dis } = item;
        return [idx, des, pred, dis] // [编号，目的地，预测点，误差距离]
      })
      setPredicts(data);
    }
    fetchData();
  }, [curShowTrajId])
  // 预测
  const { predictDispatch } = useShowPredict(chart, selectedTraj, nums, predicts, { drawOD, drawTraj, drawCurpt });
  // ---------------------------------------------

  // ---------------------------------------------
  // poi 查询
  const {
    poiDisabled,
    setPoiDisabled,
    poiState,
    poiDispatch,
    searchCompleteResult,
  } = usePoiSearch(bmap, selectedTraj);
  // ---------------------------------------------


  // 当前轨迹(速度/转向角)图层展示
  // const spdLayerData = useFeatureLayer(chart, selectedTraj, 'spd', '速度热力图层');
  // const azmLayerData = useFeatureLayer(chart, selectedTraj, 'azimuth', '转向角热力图层');


  return (
    <>
      {/* bmap 容器 */}
      <div
        key={'3-1'}
        ref={ref}
        className='bmap-container'
      ></div>
      {/* Left-Drawer */}
      <Drawer
        render={(isVisible) => (
          <Foobar
            // 抽屉是否可视
            isVisible={isVisible}
            // 预测
            onPredictDispatch={predictDispatch}
            // poi 查询
            poi={poiDisabled} // 是否开启poi查询
            onPoi={setPoiDisabled} // 开启/关闭 poi 查询
            poiField={poiState} // poi配置项
            setPoiField={poiDispatch} // poi配置项更新回调
            // SYZ
            chart={chart}
            selectedTraj={selectedTraj}
          />
        )}
        id={1}
        curId={drawerId}
        setCurId={setDrawerId}
        width={200}
        type='left'
      />
      <Tooltip
        top={tooltip.top}
        left={tooltip.left}
        display={tooltip.display}
      >
        {tooltipOptions[tooltip.type]?.()}
      </Tooltip>

      {/* EChart 图表 */}
      {/* 1. POI检索环形统计图 */}
      {
        poiDisabled ? (
          <Doughnut
            data={searchCompleteResult}
            autoplay={true}
            autoplayInterval={2000}
            style={{
              position: 'absolute',
              top: document.querySelector('#poi-frame').offsetTop,
              left: document.querySelector('#poi-frame').offsetLeft + document.querySelector('#poi-frame').offsetWidth,
            }}
          />
        ) : null
      }

      <ShoppingDrawer
        ShenZhen={ShenZhen}
      />
    </>
  )
}


export default withMouse(PagePredict);