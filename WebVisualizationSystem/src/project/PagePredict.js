// 第三方库
import React, { useRef, useEffect, useState, useContext } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// 组件
import { Drawer } from 'antd';
import SingleTrajSelector from '@/components/pagePredict/SingleTrajSelector'
import TimeSelector from '@/components/pagePredict/TimeSelector';
import TransferSelector from '@/components/pagePredict/TransferSelector';
import BrushBar from '@/components/bmapBrush/BrushBar';
// 自定义
import { getOrg, getDest } from '@/network';
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

  // org / dest color
  const orgColor = '#00FFFF';
  const destColor = '#FF0033';
  // org / dest: { {id: number, coord: number[], count: number}[] }
  const { data: org, isComplete: orgSuccess } = useReqData(getOrg);
  const { data: dest, isComplete: destSuccess } = useReqData(getDest);
  // org / dest selected res: {{org: number[], dest: number[]}}
  const [selected, setSelected] = useState({});

  // select by time
  const [byTime, setByTime] = useState([]);
  // select specific one deeply
  const [bySelect, setBySelect] = useState(-1); // 进一步筛选




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
        constantSpeed: 40,
        show: true,
        trailLength: 0.8,
        symbolSize: 5,
      },
      zlevel: 80,
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
      },
      // 标签视觉引导线
      labelLine: {
        show: false,
        showAbove: true,
        smooth: .5,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 81,
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
      },
      // 标签视觉引导线
      labelLine: {
        show: false,
        showAbove: true,
        smooth: .5,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
      zlevel: 82,
    },
    // global static trajectories
    ...globalStaticTraj,
    ...globalDynamicTraj,
    ]
  }


  useEffect(() => {
    // 实例化 chart
    chart = echarts.init(ref.current);
    chart.setOption(option);

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
        name: 'O聚类热力图',
        data,
      }]
    })
  }, [dest, destSuccess])
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
          data: org.filter(
            item => [...new Set([...sorg, ...sdest])].includes(item.id)
          ).map(item => item.coord)
        }, {
          // selected dest
          name: '筛选终点',
          data: dest.filter(
            item => [...new Set([...sorg, ...sdest])].includes(item.id)
          ).map(item => item.coord)
        }]
      })
    }
  }, [selected, org, dest])
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
      }]
    })
  }


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

  // 时间筛选静态轨迹 & 绘制
  useEffect(() => {
    chart.setOption({
      series: [{
        name: '轨迹时间筛选',
        data: (bySelect === -1) ? byTime.map(item => item.data) : [],
      }]
    })
  }, [byTime, bySelect])

  // 选择单条轨迹并绘制
  // bySelect - 选择轨迹对应的索引
  useExceptFirst(([bySelect, byTime]) => {
    let res = byTime.find(item => item.id === bySelect);
    res = res ? res.data : undefined;

    console.log(byTime);
    // res === undefined 表示没有找到数据
    if (!res) {
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
        }]
      })
    } else {
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
        }, {
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
  }, bySelect, byTime)

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
        width={'16rem'}
        // 是否显示遮罩层
        mask={false}
        // 点击蒙版是否允许关闭
        maskClosable={false}
        style={{
          position: 'absolute',
        }}
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
          onClear={() => setBySelect(-1)}
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
