// 第三方库
import React, { useRef, useEffect, useState, useReducer, useCallback } from 'react';
import * as echarts from 'echarts'; // ECharts
import 'echarts/extension/bmap/bmap';
import _ from 'lodash'; // lodash
import axios from 'axios';
import eventBus, { HISTACTION } from '@/app/eventBus';
// 通用函数
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom'; // 自主聚焦视野
import transcoords from '@/common/func/transcoords'; // 坐标纠偏
import { withMouse } from '@/components/drawer/withMouse'; // 高阶函数-监听鼠标位置
// 逻辑分离
import { useCreate } from '@/project/predict/function/useCreate';
import { usePoiSearch } from '@/project/predict/function/usePoiSearch'; // poi 查询
import { usePredict } from '@/project/predict/function/usePredict'; // 轨迹预测
import { useFeatureLayer } from '@/project/predict/function/useFeatureLayer'; // 特征热力图层展示
import { useSingleTraj } from '@/project/predict/function/useSingleTraj'; // 从候选轨迹中选择一条轨迹
// 通用组件
import Drawer from '@/components/drawer/Drawer'; // 抽屉
// 自定义组件
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

  // 当前展开的抽屉 id
  const [drawerId, setDrawerId] = useState(1);

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
    switch (type) {
      case 'showOrg':
        return {
          ...state,
          type: 'org', // 标记tooltip触发类型
          top: payload.top,
          left: payload.left,
          display: payload.display,
          data: payload.data || null,
        };
      case 'showDest':
        return {
          ...state,
          type: 'dest',
          top: payload.top,
          left: payload.left,
          display: payload.display,
          data: payload.data || null,
        }
      case 'showCur':
        return {
          ...state,
          type: 'cur',
          top: payload.top,
          left: payload.left,
          display: payload.display,
          data: payload.data || null,
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
  }
  // 添加 EChart 事件监听
  useEffect(() => {
    if (chart) {
      chart.on('click', { seriesName: '出发地' }, function (params) {
        tooltipDispatch({
          type: 'showOrg',
          payload: {
            type: 'org',
            top: newProps.current.position.top + 'px',
            left: newProps.current.position.left + 'px',
            display: '',
            data: params.data,
          }
        })
        setTimeout(() => {
          chart.on('mouseout', { seriesName: '出发地' }, function fn(params) {
            tooltipDispatch({ type: 'hidden' });
            chart.off('mouseout', fn);
          });
        }, 0)
      });

      chart.on('click', { seriesName: '目的地' }, function (params) {
        tooltipDispatch({
          type: 'showDest',
          payload: {
            type: 'dest',
            top: newProps.current.position.top + 'px',
            left: newProps.current.position.left + 'px',
            display: '',
            data: params.data,
          }
        })
        setTimeout(() => {
          chart.on('mouseout', { seriesName: '目的地' }, function fn(params) {
            tooltipDispatch({ type: 'hidden' });
            chart.off('mouseout', fn);
          });
        })
      });

      chart.on('click', { seriesName: '当前点' }, function (params) {
        tooltipDispatch({
          type: 'showCur',
          payload: {
            type: 'cur',
            top: newProps.current.position.top + 'px',
            left: newProps.current.position.left + 'px',
            display: '',
            data: params.data,
          }
        })
      });
      chart.on('mouseout', { seriesName: '当前点' }, function (params) {
        tooltipDispatch({ type: 'hidden' });
      });
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

  // 轨迹预测: 开始 / 暂停 / 清除
  const { predictDispatch } = usePredict(chart, selectedTraj, { drawOD, drawTraj, drawCurpt });

  // poi 查询
  const {
    poiDisabled,
    setPoiDisabled,
    poiState,
    poiDispatch,
    searchCompleteResult,
  } = usePoiSearch(bmap, selectedTraj);


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