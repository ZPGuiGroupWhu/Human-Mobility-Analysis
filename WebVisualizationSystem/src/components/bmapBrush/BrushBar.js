import React, { useState, useEffect } from 'react';
import IconBtn from '@/components/IconBtn.js';
import {
  brushBlack,
  brushWhite,
  clearBlack,
  clearWhite,
} from '@/icon';
import './BrushBar.scss';
import BMap from 'BMap';
import BMapLib from 'BMapLib';


let drawingManager = null;
export default function BrushBar(props) {
  const { map, data, getSelected, onClear = null } = props;

  const [state, setState] = useState(false); // 框选开关
  const [overLayers, setOverLayers] = useState([]); // 覆盖物数组
  const [selected, setSelected] = useState({}); // 选择数组



  useEffect(() => {
    const arr = [map, data];
    if (arr.includes(null) || arr.includes(undefined)) return () => { };
    // 样式
    const styleOptions = {
      strokeColor: "#00FFFF",    //边线颜色。
      fillColor: "#66FFFF",      //填充颜色。当参数为空时，圆形将没有填充效果。
      strokeWeight: 2,       //边线的宽度，以像素为单位。
      strokeOpacity: 0.8,    //边线透明度，取值范围0 - 1。
      fillOpacity: 0.6,      //填充的透明度，取值范围0 - 1。
      strokeStyle: 'solid' //边线的样式，solid或dashed。
    }
    // 实例化鼠标绘制工具
    // Demo See https://lbsyun.baidu.com/jsdemo.htm#f0_7
    // BMapLib.DrawingManager API See http://api.map.baidu.com/library/DrawingManager/1.4/docs/symbols/BMapLib.DrawingManager.html
    drawingManager = new BMapLib.DrawingManager(map, {
      isOpen: false, //是否开启绘制模式
      enableDrawingTool: false, //是否显示工具栏
      circleOptions: styleOptions, //圆的样式
      polylineOptions: styleOptions, //线的样式
      polygonOptions: styleOptions, //多边形的样式
      rectangleOptions: styleOptions //矩形的样式
    });

    drawingManager.addEventListener('overlaycomplete', (e) => {
      // 记录覆盖物
      setOverLayers(prev => ([...prev, e.overlay]));
      // 获取选框内点信息
      // See https://www.cnblogs.com/zdd2017/p/13495908.html
      const pts = e.overlay.getPath();
      const leftTop = pts[3];
      const rightBottom = pts[1];
      const ptLeftTop = new BMap.Point(leftTop.lng, leftTop.lat);
      const ptRightBottom = new BMap.Point(rightBottom.lng, rightBottom.lat);
      const bound = new BMap.Bounds(ptLeftTop, ptRightBottom);
      for (let [key, value] of Object.entries(data)) {
        for (let [idx, [lng, lat]] of Object.entries(value)) {
          let pt = new BMap.Point(lng, lat);
          if (BMapLib.GeoUtils.isPointInRect(pt, bound)) {
            // 存储对应索引，方便查找
            // 此处返回一个新数组，触发 Object.is() 浅比较
            setSelected(prev => ({
              ...prev,
              [key]: prev.hasOwnProperty(key) ? [...prev[key], idx] : [idx]
            }))
          }
        }
      }
    })
  }, [map, data])

  useEffect(() => {
    getSelected(selected);
  }, [selected])


  // 清除覆盖物
  function clearOverLayers() {
    for (let i = 0; i < overLayers.length; i++) {
      map.removeOverlay(overLayers[i]);
    }
    setOverLayers([]);
    setSelected({});
  }

  return (
    <div className='brush-bar-container'>
      <IconBtn
        imgSrc={brushWhite}
        clickCallback={() => setState(prev => {
          console.log(prev);
          if (prev) {
            // 若当前为开启状态，则后续操作将其关闭
            // 只关闭一次没效果，原因不明
            drawingManager.close();
            drawingManager.open();
            drawingManager.close();
          } else {
            drawingManager.setDrawingMode('rectangle'); // 矩形
            drawingManager.open();
          }
          return !prev;
        })}
      />
      <IconBtn
        imgSrc={clearWhite}
        clickCallback={() => {
          // 清除覆盖物
          clearOverLayers();
          onClear && onClear();
        }}
      />
    </div>
  )
}