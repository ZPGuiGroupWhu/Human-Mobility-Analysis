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
  legend: [],
  animation: false,
  visualMap: [
    {
      // 速度热力图层 visualMap
      type: 'continuous',
      min: 0, // 视觉映射最小值
      max: 40, // 视觉映射最大值
      seriesIndex: 0, // 映射的数据
      dimension: 2, // 映射数据的维度
      inRange: {
        color: ['#71ae46', '#96b744', '#c4cc38', '#ebe12a', '#eab026', '#e3852b', '#d85d2a', '#ce2626', '#ac2026'],
      },
      // 颜色条控件
      show: false,
      calculable: true, // 拖拽手柄显示
      handleSize: '100%', // 手柄大小
      itemWidth: 20, // 宽度
      itemHeight: 40, // 高度
      left: 'left',
      top: 'top',
    },
    {
      // 转向角热力图层 visualMap
      type: 'continuous',
      min: 0, // 视觉映射最小值
      max: 2, // 视觉映射最大值
      seriesIndex: 1, // 映射的数据
      dimension: 2, // 映射数据的维度
      inRange: {
        // color: ['#71ae46', '#96b744', '#c4cc38', '#ebe12a', '#eab026', '#e3852b', '#d85d2a', '#ce2626', '#ac2026'],
        color: 'red',
        colorAlpha: 0.2,
        symbolSize: [2, 10]
      },
      // 颜色条控件
      show: false,
      calculable: true, // 拖拽手柄显示
      handleSize: '100%', // 手柄大小
      itemWidth: 20, // 宽度
      itemHeight: 40, // 高度
      left: 'left',
      top: 'top',
    },
  ],
  series: [
    {
      // 利用密集轨迹点表示
      name: '速度热力图层',
      type: 'scatter',
      coordinateSystem: 'bmap',
      data: [],
      symbolSize: 2, // 像素大小
      z: 4,
    },
    {
      // 利用密集轨迹点表示
      name: '转向角热力图层',
      type: 'scatter',
      coordinateSystem: 'bmap',
      data: [],
      symbolSize: 2, // 像素大小
      z: 3,
    },
    {
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
      z: 2,
    }, {
      name: '动态单轨迹',
      type: "lines",
      coordinateSystem: "bmap",
      polyline: true,
      data: [],
      animation: false,
      lineStyle: {
        width: 0,
        color: '#00f7ff',
        cap: 'round',
        join: 'round',
      },
      effect: {
        constantSpeed: 100,
        show: true,
        trailLength: 0.3,
        symbolSize: 5,
      },
      zlevel: 1,
    }, {
      name: '出发地',
      type: 'effectScatter',
      showEffectOn: 'render', // 何时显示动效：render - 绘制完成后，emphasis - 高亮显示
      rippleEffect: {
        period: 4, // 动效周期
        scale: 3, // 波纹缩放比例
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      // 文本标签
      label: {
        show: false,
        position: 'top',
        distance: 5,
        formatter: '{a}',
        color: '#fff',
        offset: [20, -10],
      },
      // 标签视觉引导线
      labelLine: {
        show: false,
        showAbove: true,
        smooth: .1,
        length2: 20,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
    }, {
      name: '目的地',
      type: 'effectScatter',
      showEffectOn: 'render',
      rippleEffect: {
        period: 4,
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      label: {
        show: false,
        position: 'top',
        distance: 5,
        formatter: '{a}',
        color: '#fff',
        offset: [20, -10],
      },
      labelLine: {
        show: false,
        showAbove: true,
        smooth: .1,
        length2: 20,
      },
      data: [],
    }, {
      name: '当前点',
      type: 'effectScatter',
      showEffectOn: 'render',
      rippleEffect: {
        period: 4,
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 5,
      data: [],
    }, {
      name: '当前预测点',
      type: 'effectScatter',
      showEffectOn: 'render',
      rippleEffect: {
        period: 4,
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      itemStyle: {
        color: destColor,
      },
      data: [],
    }, {
      // 历史预测点集合
      name: '历史预测点',
      type: 'scatter',
      coordinateSystem: 'bmap',
      symbolSize: 8,
      itemStyle: {
        color: destColor,
      },
      // 若存在多个点，请在 data 传参时传入 color
      data: [],
    }, {
      // 历史预测点集合的路径
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
    }, {
      name: '高亮点',
      type: 'effectScatter',
      showEffectOn: 'render',
      rippleEffect: {
        period: 4,
        scale: 3,
      },
      coordinateSystem: 'bmap',
      symbolSize: 8,
      data: [],
    }, {
      name: '前N天历史静态多轨迹',
      type: 'lines',
      coordinateSystem: 'bmap',
      polyline: true,
      data: [],
      silent: true,
      lineStyle: {
        color: '#FFED0F',
        opacity: 0.8,
        width: 3,
        cap: 'round',
        join: 'round',
      },
      z: 1,
    }, {
      name: '距离可视化',
      type: 'lines',
      coordinateSystem: 'bmap',
      polyline: false,
      data: [],
      lineStyle: {
        color: '#FFED0F',
        opacity: 0.8,
        width: 1,
        type: 'dashed',
        cap: 'round',
        join: 'round',
      },
    }
  ]
}