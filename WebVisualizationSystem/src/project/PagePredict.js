// 第三方库
import React, { useRef, useEffect, useState, useContext } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// 组件
import { Drawer } from 'antd';
import TrajSelector from '@/components/pagePredict/TrajSelector';
import TransferSelector from '@/components/pagePredict/TransferSelector';
import BrushBar from '@/components/bmapBrush/BrushBar';
// 自定义
import { getOrgDemo, getDestDemo, getClusterO, getClusterD } from '@/network';
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
// chart 实例
let chart = null;
export default function PagePredict(props) {
  // Context 对象
  const drawerVisibleObj = useContext(drawerVisibility);
  // 数据
  const { data: org, isComplete: orgSuccess } = useReqData(getOrgDemo);
  const { data: dest, isComplete: destSuccess } = useReqData(getDestDemo);
  // OD 聚类结果：[lng,lat,count]
  const { data: orgCluster, isComplete: orgClusterSuccess } = useReqData(getClusterO);
  const { data: destCluster, isComplete: destClusterSuccess } = useReqData(getClusterD);
  const [byTime, setByTime] = useState([]); // 依据时间筛选轨迹
  const [bySelect, setBySelect] = useState(byTime); // 进一步筛选

  const [selected, setSelected] = useState({});


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
      right: '100rem',
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
    // // https://echarts.apache.org/zh/option.html#brush
    // brush: {
    //   // toolbox 相关按钮
    //   toolbox: ['rect', 'polygon', 'keep', 'clear'],
    //   // 数据关联
    //   brushLink: [0, 1],
    //   brushType: 'rect',
    //   brushMode: 'single',
    //   brushStyle: {
    //     borderWidth: 1,
    //     color: 'rgba(120,140,180,0.3)',
    //     borderColor: 'rgba(120,140,180,0.8)'
    //   },
    //   // 防抖
    //   throttleType: 'debounce',
    //   throttleDelay: 500,
    // },
    animation: false,
    visualMap: [
      // OD Cluster Heatmap
      {
        // https://echarts.apache.org/zh/option.html#visualMap
        type: 'continuous',
        // 视觉映射定义域
        min: 0,
        max: 300,
        // 不显示 visualMap 组件
        show: true,
        left: 20,
        bottom: 10,
        // 映射维度
        dimension: 2,
        seriesIndex: [2, 3], // OD聚类热力图
        // 定义域颜色范围
        inRange: {
          color: ['blue', 'blue', 'green', 'yellow', 'red'],
        },
        textStyle: {
          color: "#fff",
        }
      },
    ],
    series: [{
      // org
      name: '起点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: '#3498DB',
      }
    }, {
      // dest
      name: '终点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'triangle',
      data: [],
      itemStyle: {
        color: '#E74C3C',
      }
    }, {
      id: 'o-heatmap',
      name: 'O聚类热力图',
      // https://echarts.apache.org/zh/option.html#series-heatmap
      type: 'heatmap',
      coordinateSystem: 'bmap',
      pointSize: 20,
      blurSize: 20,
      // 高亮状态图形样式
      emphasis: {
        // 高亮效果
        focus: 'series',
      },
      data: [],
    }, {
      id: 'd-heatmap',
      name: 'D聚类热力图',
      // https://echarts.apache.org/zh/option.html#series-heatmap
      type: 'heatmap',
      coordinateSystem: 'bmap',
      pointSize: 20,
      blurSize: 20,
      // 高亮状态图形样式
      emphasis: {
        // 高亮效果
        focus: 'series',
      },
      data: [],
    }, {
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
      // selected org
      name: '筛选起点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'circle',
      data: [],
      itemStyle: {
        color: '#3498DB',
      }
    }, {
      // selected dest
      name: '筛选终点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 5,
      symbol: 'triangle',
      data: [],
      itemStyle: {
        color: '#E74C3C',
      }
    },
    ...globalStaticTraj,
    ...globalDynamicTraj,
    ]
  }


  useEffect(() => {
    // 实例化 chart
    chart = echarts.init(ref.current);
    chart.setOption(option);
    // 显示 loading
    chart.showLoading();

    // 获取地图实例, 初始化
    bmap = chart.getModel().getComponent('bmap').getBMap();
    bmap.centerAndZoom(props.initCenter, props.initZoom);
    bmap.setMapStyleV2({
      styleId: 'f65bcb0423e47fe3d00cd5e77e943e84'
    });
  }, [])

  // 全局静态轨迹
  const t0 = useStaticTraj(chart, { ...trajColorBar[0] });
  // const t1 = useStaticTraj(chart, { ...trajColorBar[1] });
  // const t2 = useStaticTraj(chart, { ...trajColorBar[2] });
  // const t3 = useStaticTraj(chart, { ...trajColorBar[3] });
  // const t4 = useStaticTraj(chart, { ...trajColorBar[4] });
  // const t5 = useStaticTraj(chart, { ...trajColorBar[5] });

  // 全局OD
  useEffect(() => {
    if (orgSuccess) {
      // 取消 loading
      chart.hideLoading();
      chart.setOption({
        series: [{
          name: '起点',
          data: org,
        }]
      })
    }
  }, [org, orgSuccess])
  useEffect(() => {
    if (destSuccess) {
      // 取消 loading
      chart.hideLoading();
      chart.setOption({
        series: [{
          name: '终点',
          data: dest,
        }]
      })
    }
  }, [dest, destSuccess])
  useEffect(() => {
    if (orgClusterSuccess) {
      chart.hideLoading();
      chart.setOption({
        series: [{
          name: 'O聚类热力图',
          data: orgCluster,
        }]
      })
    }
  }, [orgCluster, orgClusterSuccess])
  useEffect(() => {
    if (destClusterSuccess) {
      chart.hideLoading();
      chart.setOption({
        series: [{
          name: 'D聚类热力图',
          data: destCluster,
        }]
      })
    }
  }, [destCluster, destClusterSuccess])
  // 根据筛选结果更新样式
  useEffect(() => {
    console.log(selected);
    const { org: sorg = [], dest: sdest = [] } = selected;
    const arr = [org, dest, sorg, sdest];
    if (arr.reduce((prev, cur) => {
      let res = !prev ? false : prev && (cur.length !== 0)
      return res;
    }, true)) {
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
          data: org.filter((item, idx) => sorg.includes(idx.toString()))
        }, {
          // selected dest
          name: '筛选终点',
          data: dest.filter((item, idx) => sdest.includes(idx.toString()))
        }]
      })
    }
  }, [selected, org, dest])


  // 全局静态轨迹图例
  useEffect(() => {
    const data = trajColorBar.map(item => ({
      name: item.static,
      itemStyle: {
        color: item.color,
      }
    }))
    chart.setOption({
      legend: [{
        data,
      }, {
        data: [{
          name: '起点'
        }, {
          name: '终点'
        }],
      }]
    })
  }, [])


  // 存储时间信息
  /**timer:
   * dateStart
   * dateEnd
   * hourStart
   * hourEnd
   */
  const [timer, timerDispatch] = useTime();

  // bySelect 相当于 byTime 的副本，需保持同步
  useEffect(() => {
    setBySelect(byTime);
  }, [byTime])

  // 时间筛选静态轨迹 & 绘制
  useExceptFirst((data) => {
    chart.setOption({
      series: [{
        name: '轨迹时间筛选',
        data: data.map(item => item.data),
      }]
    })
  }, bySelect)


  // 框选事件
  useEffect(() => {
    chart.on('brush', () => {
      chart.setOption({
        bmap: {
          roam: false,
        }
      })
    })

    chart.on('brushEnd', () => {
      chart.setOption({
        bmap: {
          roam: true,
        }
      })
    })
  }, [chart])


  function onClear () {
    chart && chart.setOption({
      series: [{
        // org
        name: '起点',
        itemStyle: {
          color: '#3498DB',
        }
      }, {
        // dest
        name: '终点',
        itemStyle: {
          color: '#E74C3C',
        }
      },{
        // selected org
        name: '筛选起点',
        data: []
      }, {
        // selected dest
        name: '筛选终点',
        data: []
      }]
    })
  }

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
        data={{ org, dest }}
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
        width={'24rem'}
        // 是否显示遮罩层
        mask={false}
        // 点击蒙版是否允许关闭
        maskClosable={false}
        style={{
          position: 'absolute',
        }}
      >
        <TrajSelector
          setByTime={setByTime}
          timer={timer}
          timerDispatch={timerDispatch}
        />
        <TransferSelector
          data={byTime}
          setData={setBySelect}
        />
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
