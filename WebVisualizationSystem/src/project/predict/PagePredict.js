// 第三方库
import React, { useRef, useEffect, useState, useContext, useReducer } from 'react';
import * as echarts from 'echarts'; // ECharts
import 'echarts/extension/bmap/bmap';
import _ from 'lodash'; // lodash
import { Switch, Slider, Space, Input, Radio } from 'antd'; // Ant-Design
// 通用函数
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom'; // 自主聚焦视野
import transcoords from '@/common/func/transcoords'; // 坐标纠偏
import { eventEmitter } from '@/common/func/EventEmitter'; // 发布订阅
import { withMouse } from '@/components/bottom-drawer/withMouse'; // 高阶函数-监听鼠标位置
// 逻辑分离
import { useCreate } from '@/project/predict/function/useCreate'; // 
import { usePoiSearch } from '@/project/predict/function/usePoiSearch'; // poi 查询
import { usePredict } from '@/project/predict/function/usePredict'; // 轨迹预测
// 通用组件
import BottomDrawer from '@/components/bottom-drawer/BottomDrawer'; // 底部抽屉
// 自定义组件
import BrushBar from '@/components/bmapBrush/BrushBar'; // 框选功能条
import MyDrawer from '@/components/drawer/MyDrawer';
import SimpleBar from '@/components/simpleBar/SimpleBar'; // 小型抽屉栏
import InfoBar from '@/components/infoBar/InfoBar'; // 框型小型抽屉栏
import Doughnut from '@/components/pagePredict/doughnut/Doughnut'; // Echarts 环形统计图
import Foobar from './components/foobar/Foobar'; // 底部功能栏
import EChartbar from './components/charts/EChartbar'; // EChart 侧边栏
import RelationChart from './components/charts/relation-chart/RelationChart'; // EChart关系折线图
import Tooltip from '@/components/tooltip/Tooltip'; // 自定义悬浮框
import ScatterTooltip from './components/scatter-tooltip/ScatterTooltip'; // 点-tooltip
// 样式
import '@/project/bmap.scss';
// 全局状态管理
import Store from '@/store';





// iconfont Symbol 在线 url
const iconScriptUrl = '//at.alicdn.com/t/font_2577661_dmweq4qmkar.js';

function PagePredict(props) {
  const { state, dispatch } = useContext(Store); // 全局状态管理 Store
  const [selectedTraj, setSelectedTraj] = useState(null); // 存放单轨迹数据
  useEffect(() => {
    if (state.selectedTraj) {
      const data = transcoords(state.selectedTraj.data); // 坐标纠偏
      const traj = _.cloneDeep(state.selectedTraj); // 深拷贝，返回 immutable 对象
      Reflect.set(traj, 'data', data);
      setSelectedTraj(traj);
    }
  }, [state.selectedTraj]);

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
      });
      chart.on('mouseout', { seriesName: '出发地' }, function (params) {
        tooltipDispatch({ type: 'hidden' });
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
      });
      chart.on('mouseout', { seriesName: '目的地' }, function (params) {
        tooltipDispatch({ type: 'hidden' });
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
            color: curColor,
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

  // 轨迹预测: 开始 / 暂停 / 清除
  const { predictDispatch } = usePredict(chart, selectedTraj, { drawOD, drawTraj, drawCurpt });













  // scatter style: org / dest color
  const orgColor = '#00FFFF'; // 起点颜色
  const destColor = '#FF0033'; // 终点颜色
  const curColor = '#E53935'; // 当前点颜色

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

  // org / dest State: 起终点是否被选中
  const [brushData, setBrushData] = useState({});
  // org / dest selected res: {{org: number[], dest: number[]}}
  const [selected, setSelected] = useState({});
  // select by brush
  const [byBrush, setByBrush] = useState([]);
  // 存储用于预测的单条轨迹
  const [usedForPredict, setUsedForPredict] = useState({});


  // POI查询-多选框-选项
  const radioOptions = [
    { label: '起点', value: 'start' },
    { label: '当前点', value: 'current' },
    { label: '终点', value: 'end' },
  ]




  // --------------------- 自定义函数 ---------------------
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




  // 依据 byBrush 筛选结果进行单轨迹动效绘制
  function singleTrajByBrush(idx) {
    const res = byBrush[idx].data;
    // 记录轨迹
    setUsedForPredict(byBrush[idx] || {});
    drawTraj(res)
    drawOD(res);
  }

  // --------------------- Hooks ---------------------



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

  // poi 查询
  const {
    poiDisabled,
    setPoiDisabled,
    poiState,
    poiDispatch,
    searchCompleteResult
  } = usePoiSearch(bmap, usedForPredict);

  return (
    <>
      {/* bmap 容器 */}
      <div
        key={'3-1'}
        ref={ref}
        className='bmap-container'
      ></div>
      {/* POI检索环形统计图 */}
      <Doughnut
        data={searchCompleteResult}
        style={{
          position: 'absolute',
          bottom: '0',
          left: '0',
          display: poiDisabled && searchCompleteResult ? '' : 'none',
        }}
      />
      {/* Bottom-Drawer */}
      <BottomDrawer render={() => (
        <Foobar
          onPredictDispatch={predictDispatch} // 预测
        />
      )} height={170} />
      {/* EChart-bar */}
      <EChartbar>
        <RelationChart
          titleText='时间 - 速度/转向角'
          legendData={['速度', '转向角']}
          xAxisData={[1, 2, 3]}
          yAxis={['速度(km/h)', '转向角(rad)']}
          data={[[1, 2, 3], [2, 3, 4]]}
        />
      </EChartbar>
      <Tooltip
        top={tooltip.top}
        left={tooltip.left}
        display={tooltip.display}
      >
        {tooltipOptions[tooltip.type]?.()}
      </Tooltip>



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
            <div style={{ width: '100px' }}>{`半径(${poiState.radius} 米)`}</div>
            <Slider
              min={1}
              max={500}
              defaultValue={poiState.radius}
              disabled={!poiDisabled}
              onChange={
                _.debounce(
                  (value) => poiDispatch({ type: 'radius', payload: value }),
                  200,
                  { trailing: true }
                )
              }
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
              onChange={
                _.debounce(
                  (e) => { poiDispatch({ type: 'keyword', payload: e.target.value }) },
                  500,
                  { trailing: true }
                )
              }
              allowClear
              size='middle'
              style={{ width: '100px' }}
            />
          </Space>
          <Radio.Group
            name='poiSearch'
            defaultValue={poiState.description}
            disabled={!poiDisabled}
            onChange={(e) => { poiDispatch({ type: 'description', payload: e.target.value }); }}
          >
            {radioOptions.map((item, idx) => {
              return (
                <Radio value={item.value} key={idx}>{item.label}</Radio>
              )
            })}
          </Radio.Group>
        </InfoBar>
      </MyDrawer>
    </>
  )
}


export default withMouse(PagePredict);