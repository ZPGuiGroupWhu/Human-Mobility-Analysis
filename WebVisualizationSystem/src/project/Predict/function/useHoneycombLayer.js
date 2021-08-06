import { useCallback, useEffect, useMemo } from 'react';

export const useHoneycombLayer = (data, bmapgl) => {
  const mapvgl = window.mapvgl;
  
  const view = useMemo(() => {
    return bmapgl ? new mapvgl.View({
      map: bmapgl
    }) : null;
  }, [bmapgl])

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
    if (!view) return () => { };
    let honeycombLayer = new mapvgl.HoneycombLayer({
      // 设置颜色梯度
      gradient: {
        0.0: 'rgb(50, 50, 256)',
        0.1: 'rgb(50, 250, 56)',
        0.5: 'rgb(250, 250, 56)',
        1.0: 'rgb(250, 50, 56)'
      },
      // 是否显示文字
      showText: true,
      // 设置文字样式
      textOptions: {
        fontSize: 13,
        color: '#fff',
        format: function (count) {
          return count >= 10000 ? Math.round(count / 1000) + 'k'
            : count >= 1000 ? Math.round(count / 100) / 10 + 'k' : count;
        }
      },
      height: 40, // 最大值的像素高度，为0时显示平面
      enableCluster: true, // 开启点聚合，建议数据量较大时打开，会提前根据地图级别将距离较近的点进行聚合
      opacity: 0.9, // 透明度
      maxZoom: 16, // 图层刷新最大地图级别，高于此值不再更新图层
      minZoom: 8, // 图层刷新最小地图级别，低于此值不再更新图层
      size: 20 // 蜂窝图像素宽度
    });
    view.addLayer(honeycombLayer);
    honeycombLayer.setData(format(data));
  }, [view, data])

  return view;
}