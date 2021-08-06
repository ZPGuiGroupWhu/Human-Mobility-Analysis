import { useEffect, useMemo, useCallback } from 'react';

export const useHeatGridLayer = (data, bmapgl) => {
  const mapvgl = window.mapvgl;

  // 创建图层管理器
  const view = useMemo(() => (
    bmapgl ? new mapvgl.View({
      effects: [new mapvgl.BrightEffect({
        threshold: .5,
        blurSize: 2,
        clarity: 0.8
      })],
      map: bmapgl
    }) : null
  ), [bmapgl])

  const format = useCallback((data)=>{
    return data.map(item => (
      {
        geometry: {
          type: 'Point',
          coordinates: item
        }, 
        properties: {
          count: 1,
        }
      }
    ))
  }, [])

  useEffect(() => {
    if (!view || !data.length) return () => { };

    let layer = new mapvgl.HeatGridLayer({
      max: 80, // 最大阈值
      min: 0, // 最小阈值
      // color: function() {
      //     return 'rgb(200, 255, 0)';
      // },
      gridSize: 300,
      // style: 'normal',
      gradient: { // 对应比例渐变色
        0: 'rgb(50, 50, 256)',
        0.5: 'rgb(178, 202, 256)',
        1: 'rgb(250, 250, 256)'
      },
      // textOptions: {
      //     show: true,
      //     color: '#f00'
      // },
      riseTime: 1800, // 楼块初始化升起时间
      maxHeight: 3000, // 最大高度
      minHeight: 0 // 最小高度
    });
    view.addLayer(layer);
    layer.setData(format(data));
  }, [view, data])

  return view;
}