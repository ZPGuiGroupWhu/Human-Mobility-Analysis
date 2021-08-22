// 第三方库
import React, { useRef, useEffect, useState, useContext, useReducer } from 'react';
import * as echarts from 'echarts'; // ECharts
import 'echarts/extension/bmap/bmap'; // ECharts
import _ from 'lodash'; // lodash
import { Drawer, Select, message, Switch, Slider, Space, Input, Radio } from 'antd'; // Ant-Design
// 通用函数
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom';
import { eventEmitter } from '@/common/func/EventEmitter';
// 通用 Hooks
import { useExceptFirst } from '@/common/hooks/useExceptFirst';
import { useTime } from '@/common/hooks/useTime';
// 逻辑分离
import { useCreate } from '@/project/predict/function/useCreate'; // 
import { usePoiSearch } from '@/project/predict/function/usePoiSearch'; // poi 查询
import { useOD } from '@/project/predict/function/useOD'; // OD 显示 & heatmap 显示 & 图例显示
import { usePredict } from '@/project/predict/function/usePredict'; // 轨迹预测
// 网络请求
import { getOrg, getDest, selectByTime } from '@/network';
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
import Doughnut from '@/components/pagePredict/doughnut/Doughnut'; // Echarts 环形统计图
// Context 对象导入
import { drawerVisibility } from '@/context/mainContext';
// 样式
import '@/project/bmap.scss';
// 全局状态管理
import Store from '@/store';





// iconfont Symbol 在线 url
const iconScriptUrl = '//at.alicdn.com/t/font_2577661_dmweq4qmkar.js';

export default function PagePredict(props) {
  // 全局状态管理 Store
  const value = useContext(Store);
  useEffect(()=>{
    console.log(value);
  }, [])

  // Context 对象
  const drawerVisibleObj = useContext(drawerVisibility);


  // scatter style: org / dest color
  const orgColor = '#00FFFF'; // 起点颜色
  const destColor = '#FF0033'; // 终点颜色
  const curColor = '#E53935'; // 当前点颜色

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
      default:
        return;
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


  // POI查询-多选框-选项
  const radioOptions = [
    { label: '起点', value: 'start' },
    { label: '当前点', value: 'current' },
    { label: '终点', value: 'end' },
  ]

  // 容器 ref 对象
  const ref = useRef(null);


  // --------------------- 自定义函数 ---------------------
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
      // 请求所有轨迹时，将单选的轨迹记录清除
      setBySelect(-1);
      selectByTime(start, end).then(
        res => {
          console.log(res);
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
  }

  // --------------------- Hooks ---------------------
  // 首次进入页面，创建 echarts 实例
  const initCenter = props.initCenter;
  const initZoom = props.initZoom;
  const { bmap, chart } = useCreate({ ref, initCenter, initZoom })

  // 轨迹预测: 开始 / 暂停 / 清除
  const { predictDispatch } = usePredict(chart, usedForPredict, { singleOD, singleTraj, singleCurpt, clearSingleTraj })

  // OD 显示 & heatmap 显示 & 图例显示
  const { setodShow, setHeatmapShow } = useOD(org, dest, ctrlState.legend, bmap, chart);

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
