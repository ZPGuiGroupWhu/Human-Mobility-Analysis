import React, { useState, useEffect, useRef } from 'react';
import _ from 'lodash';
import './CharacterWindow.scss';
// ECharts
import * as echarts from 'echarts';

let myChart = null;

export default function ParallelChart(props) {
  // Echarts 容器实例
  const ref = useRef(null);
  // 获取chart数据
  const {
    returnSelectedResult,
    data,
    characterReload,
    userId
  } = props;

  // 特征属性
  const characters = [
    { dim: 0, name: '移动总距离' },
    { dim: 1, name: '单次最大距离'},
    { dim: 2, name: '平均速度' },
    { dim: 3, name: '最大速度' },
    { dim: 4, name: '最大转向角' }
  ];

  // 刷选时更新characterSelected数组
  function onAxisAreaSelected(params) {
    let series0 = myChart.getModel().getSeries()[0];
    let indices0 = series0.getRawIndicesByActiveState('active');
    const payload = indices0.map(item => {
      let trajId = [userId, item].join('_'); // 字符串拼接得到轨迹编号
      return trajId;
    });
    // 针对api自带的清除工具 如果清空 则返回所有的轨迹编号求交集，反之返回选择的轨迹编号
    returnSelectedResult(payload.length === 0? '' : payload)
  };

  // 选框样式
  const areaSelectStyle = {
    width: 15,
    borderWidth: .8,
    borderColor: 'rgba(160,197,232)',
    color: 'rgba(160,197,232)',
    opacity: .4,
  }

  // 线样式
  const lineStyle = {
    width: 1,
    opacity: 0.5,
    cap: 'round',
    join: 'round',
  };

  const option = {
    // 工具栏配置
    toolbox: {
      feature: {
        brush: {
          title: {
            clear: '还原',
          },
          // 更换icon图标
          icon: {
            clear: "M819.199242 238.932954l136.532575 0c9.421335 0 17.066098-7.644762 17.067994-17.066098l0-136.532575-34.134092 0L938.665719 174.927029C838.326316 64.646781 701.016372 0 563.20019 0 280.88245 0 51.20019 229.682261 51.20019 512s229.682261 512 512 512c160.289736 0 308.325479-72.977903 406.118524-200.225656l-27.067616-20.78799c-91.272624 118.749781-229.445258 186.879554-379.050907 186.879554-263.509197 0-477.865908-214.356711-477.865908-477.865908S299.689097 34.134092 563.20019 34.134092c131.090991 0 262.003755 63.224764 356.406712 170.664771l-100.405764 0L819.201138 238.932954z",
          }
        }
      },
      iconStyle: {
        color: '#fff', // icon 图形填充颜色
        borderColor: '#fff', // icon 图形描边颜色
      },
      emphasis: {
        iconStyle: {
          color: '#7cd6cf',
          borderColor: '#7cd6cf',
        }
      }
    },
    // 框选工具配置
    brush: {
      toolbox: ['clear'],
      xAxisIndex: 0,
      throttleType: 'debounce',
      throttleDelay: 300,
    },
    visualMap: [
      {
        type: 'continuous',
        min: 0,
        max: 30,
        splitNumber: 10,
        show: false,
        dimension: 1,
      }
    ],
    parallelAxis: characters.map((item) => ({
      dim: item.dim,
      name: item.name,
      areaSelectStyle: areaSelectStyle
    })),
    parallel: {
      left: '10%',
      right: '13%',
      bottom: '15%',
      top: '5%',
      parallelAxisDefault: {
        type: 'value',
        nameLocation: 'start',
        nameGap: 10,
        nameTextStyle: {
          fontSize: 12,
          color: '#fff',
        },
        min: 'dataMin',
        max: 'dataMax',
        axisLine: {
          lineStyle: {
            color: '#fff',
          }
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          formatter: (value) => (parseInt(value)),
        },
      }
    },
    series: [
      {
        name: '特征筛选',
        type: 'parallel',
        lineStyle: lineStyle,
        inactiveOpacity: 0.02,
        activeOpacity: 1,
        realtime: true,
        data: [],
      },
    ]
  };

  // 初始化 ECharts 实例对象
  useEffect(() => {
    if (!ref.current) return () => { };
    myChart = echarts.init(ref.current);
    myChart.setOption(option);
    myChart.on('axisareaselected', onAxisAreaSelected);
    window.onresize = myChart.resize;
  }, [ref])

  // 当 data改变或者 finalSelected改变时
  useEffect(() => {
    setTimeout(() => {
      myChart?.setOption({
        series: [{
          name: '特征筛选',
          data: data
        }]
      })
    }, 100)
  }, [data])

  return (
    <div className='character-parallel'
      ref={ref}
    ></div>
  )
}