import { useEffect, useState, useMemo } from 'react';

export const useCurve = (org, dest, bmapgl) => {
  const mapvgl = window.mapvgl;
  const [data, setData] = useState(null);

  let curve = new mapvgl.BezierCurve();

  // 创建图层管理器
  const view = useMemo(() => (
    bmapgl ? new mapvgl.View({
      effects: [
        new mapvgl.BrightEffect({
          threshold: 0.5, // 发光门槛阈值，范围0.0~1.0，值越低发光效果越亮
          blurSize: 2, // 模糊值半径
          clarity: 0.8, // 清晰度
        }),
      ],
      map: bmapgl
    }) : null
  ), [bmapgl])

  // 规范化数据
  useEffect(() => {
    let data = [];
    const lens = org.length;
    for (let i = 0; i < lens; i++) {
      curve.setOptions({
        start: org[i],
        end: dest[i],
      });
      let curveData = curve.getPoints(80);
      data.push({
        geometry: {
          type: 'LineString',
          coordinates: curveData,
        },
        properties: {
          count: Math.random()
        }
      })
    }

    setData(data);
  }, [org, dest])

  // (配置 ｜ 添加 ｜ 管理)飞线图层
  useEffect(() => {
    if (!view || !data.length) return () => { };
    let aniLineLayer = new mapvgl.LineTripLayer({
      color: 'rgb(255, 255, 204)', // 飞线动画颜色
      step: 0.3
    });
    view.addLayer(aniLineLayer);
    aniLineLayer.setData(data);

    let lineLayer = new mapvgl.SimpleLineLayer({
      blend: 'lighter',
      color: 'rgb(255, 153, 0, 0.8)' // 飞线颜色
    });
    view.addLayer(lineLayer);
    lineLayer.setData(data);
  }, [view, data])

  return view;
}