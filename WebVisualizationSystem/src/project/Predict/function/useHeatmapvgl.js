import { useEffect, useMemo } from 'react';
import transcoords from '@/common/func/transcoords';

export const useHeatmapvgl = (data, bmapgl) => {
  const mapvgl = window.mapvgl;

  const view = useMemo(() => (
    bmapgl ? new mapvgl.View({
      effects: [
        new mapvgl.BloomEffect({
          threshold: 0.2,
          blurSize: 2.0
        }),
      ],
      map: bmapgl
    }) : null
  ), [bmapgl])

  // 规范化数据
  function heatmap(data) {
    return data.map(item => ({
      geometry: {
        type: 'Point',
        coordinates: item,
      },
      properties: {
        count: 1,
      }
    }))
  }

  // heatmapLayer
  useEffect(() => {
    if (!view) return () => { };
    let layer = new mapvgl.HeatmapLayer({
      size: 20, // 单个点绘制大小
      height: 80, // 最大高度，默认为0
      unit: 'px', // 单位，m:米，px: 像素
      gradient: {
        0.25: 'rgba(89, 233, 179, 1)',
        0.55: 'rgba(182, 243, 147, 1)',
        0.85: 'rgba(254, 255, 140, 1)',
        0.9: 'rgba(217, 29, 28, 1)',
      },
    });
    let res = heatmap(data);
    view.addLayer(layer);
    layer.setData(res);
  }, [view]);

  // heatPointLayer
  useEffect(() => {
    if (!view) return () => { };
    let layer = new mapvgl.HeatPointLayer({
      blend: 'lighter',
      style: 'grid',
      size: 4,
      min: 0,
      max: 1000,
      gradient: {
        0: 'rgb(200, 200, 200, 0)',
        0.2: 'rgb(200, 200, 200, 0)',
        0.5: 'rgb(226, 95, 0, 0.9)',
        1: 'rgb(239, 209, 19, 0.9)'
      }
    });
    let res = heatmap(data);
    view.addLayer(layer);
    layer.setData(res);
  }, [view])

  return view;
}