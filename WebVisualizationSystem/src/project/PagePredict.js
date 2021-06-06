// 第三方库
import React, { useState, useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// 组件
import Sider from '@/components/sider/Sider';
// 自定义
import { getOrgDemo, getDestDemo } from '@/network';
import { useReqData } from '@/common/hooks/useReqData';
import { useStaticTraj } from '@/common/hooks/useStaticTraj';
import { globalStaticTraj, globalDynamicTraj } from '@/common/options/globalTraj';
import { trajColorBar } from '@/common/options/trajColorBar';
// 样式
import './bmap.scss';





// 地图实例
let bmap = null;
// chart 实例
let chart = null;
export default function PagePredict(props) {
  // 数据
  const { data: org, isComplete: orgSuccess } = useReqData(getOrgDemo);
  const { data: dest, isComplete: destSuccess } = useReqData(getDestDemo);
  // 容器 ref 对象
  const ref = useRef(null);
  // 历史 option 配置记录
  const [histOption, setHistOption] = useState(null);

  // 静态配置项
  const option = {
    bmap: {
      center: [120.13066322374, 30.240018034923],
      zoom: 12,
      minZoom: 0,
      maxZoom: 20,
      roam: true,
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
    },{
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
    // visualMap: [{
    //   type: 'piecewise',
    //   // 自动分段 - 段数
    //   splitNumber: 10,
    //   // 定义域颜色范围
    //   inRange: ['#fff59d', '#c5e1a5', '#80deea', '#90caf9', '#ce93d8', '#ef9a9a'],
    //   // 作用的数据索引
    //   seriesIndex: [2, 3], // 全局轨迹 + 全局轨迹动画
    //   // 作用的数据维度
    //   dimension: 2,
    // }],
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
    })
  }, [])

  // 全局静态轨迹
  const t0 = useStaticTraj(chart, { ...trajColorBar[0] });
  const t1 = useStaticTraj(chart, { ...trajColorBar[1] });
  const t2 = useStaticTraj(chart, { ...trajColorBar[2] });
  const t3 = useStaticTraj(chart, { ...trajColorBar[3] });
  const t4 = useStaticTraj(chart, { ...trajColorBar[4] });
  const t5 = useStaticTraj(chart, { ...trajColorBar[5] });

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

  // 全局静态轨迹图例
  useEffect(()=>{
    const data = trajColorBar.map(item => ({
      name: item.static,
      itemStyle: {
        color: item.color,
      }
    }))
    chart.setOption({
      legend: [{
        data,
      },{
        data: [{
          name: '起点'
        },{
          name: '终点'
        }],
      }]
    })
  })






  return (
    <>
      <div
        key={'3-1'}
        ref={ref}
        className='bmap-container'
      ></div>
      <Sider key={'3-2'} floatType='left'>PagePredict</Sider>
      <Sider key={'3-3'} floatType='right'>PagePredict</Sider>
    </>
  )
}
