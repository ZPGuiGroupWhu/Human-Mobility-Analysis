const orgColor = '#00FFFF'; // 起点颜色
const destColor = '#FF0033'; // 终点颜色
const curColor = '#E53935'; // 当前点颜色

// ECharts 静态配置项
export const option = {
  bmap: {
    center: [120.13066322374, 30.240018034923],
    zoom: 12,
    minZoom: 0,
    maxZoom: 20,
    roam: true, // 若设为 false，则地图底图不可移动或缩放
    // mapStyle: bmapStyle,
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