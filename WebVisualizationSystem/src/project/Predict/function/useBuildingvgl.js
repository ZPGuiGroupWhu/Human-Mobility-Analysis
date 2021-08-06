import { useEffect, useRef, useMemo } from 'react';
import transcoords from '@/common/func/transcoords';
import axios from 'axios';
import texture from '@/assets/light.jpg';

export const useBuildingvgl = (bmapgl) => {
  const mapvgl = window.mapvgl;

  const view = useMemo(() => (
    bmapgl ? new mapvgl.View({
      effects: [
        // 创建炫光处理通道
        new mapvgl.BloomEffect({
          threshold: 0.8, // 0 ~ 1，值越低月亮
          blurSize: 2 // 炫光模糊度
        }),
      ],
      map: bmapgl
    }) : null
  ), [bmapgl])

  // 缓存建筑矢量数据
  const polygons = useRef([]);
  useEffect(() => {
    if (!view || polygons.current.length) return () => { };
    // 请求建筑物 json 数据
    axios.get(
      process.env.PUBLIC_URL + '/shenzhen/building.json',
      {
        baseURL: 'http://localhost:3000',
        timeout: 2000,
      }
    ).then(res => {
      // console.log(res); 
      return res.data.geometries
    }).then(res => {
      const lens = res.length;
      for (let i = 0; i < lens; i++) {
        let pts = res[i].coordinates[0];
        pts = transcoords(pts);
        // 转换格式
        polygons.current.push({
          geometry: {
            type: 'Polygon',
            coordinates: [pts]
          },
          properties: {
            height: Math.random() * 100
          }
        });
      }
    }).then(() => {
      let layer = new mapvgl.ShapeLayer({
        options: {
          renderOrder: 10, // 渲染顺序
        },
        texture, // 纹理贴图
        isTextureFull: true,
        blend: 'lighter',
        color: [0.8, 0.8, 0.1],
        opacity: 1.0,
        riseTime: 1000, // 建筑升起动画
        style: 'windowAnimation', // 窗户动画样式
      });

      view.addLayer(layer); // 添加图层到图层管理器
      layer.setData(polygons.current); // 展示数据
    }).catch(err => {
      console.log(err);
    })
  }, [view])

  return view;
}