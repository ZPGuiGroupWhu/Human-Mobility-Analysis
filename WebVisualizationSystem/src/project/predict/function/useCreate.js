import { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts'; // ECharts
import 'echarts/extension/bmap/bmap'; // ECharts
// 配置项
import { option } from '@/project/predict/config/option.js'; // Echarts 静态配置项

export const useCreate = ({ ref, initCenter, initZoom }) => {
  // echarts 实例对象
  const [chart, setChart] = useState(null);
  // bmap 底图实例
  const bmap = useRef(null);

  // 获取 bmap 实例
  function getBMapInstance(chart = null) {
    try {
      if (!chart) throw new Error('echarts实例不存在');
      // 获取地图实例, 初始化
      let bmap = chart.getModel().getComponent('bmap').getBMap();
      bmap.setMapStyleV2({
        styleId: 'f65bcb0423e47fe3d00cd5e77e943e84'
      });
      return bmap;
    } catch (err) {
      console.log(err);
    }
  }

  // 首次进入页面，创建 echarts 实例
  useEffect(() => {
    // 实例化 chart
    setChart(() => {
      const chart = echarts.init(ref.current);
      chart.setOption(option);
      bmap.current = getBMapInstance(chart);
      // bmap.current.centerAndZoom(initCenter, initZoom);
      return chart;
    });
  }, [])

  return {
    bmap: bmap.current,
    chart,
  }
}