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



/**
 * @param {object} map - 百度地图实例
 * @param {{ org:{id: number, coord: number[], count: number}[], dest:{id: number, coord: number[], count: number}[]}} data - 坐标数组
 * @param {function} getSelected - 提供了 Brush 向外暴露框选数据的接口
 * @param {function} onClear - 清除选框时额外的操作
 */
let drawingManager = null;
export default function BrushBar(props) {
  const { map, data, getSelected, onClear = null } = props;

  const [state, setState] = useState(false); // 框选开关
  const [overLayers, setOverLayers] = useState([]); // 覆盖物数组
  const [selected, setSelected] = useState({}); // 选择数组

  /**
   * 通过起点与终点经纬度关系判断筛选框绘制的方向
   * @param {number[][]} rectPts - 筛选框的四个角，顺序从起点开始顺时针排布
   */
  function paintDirection(rectPts) {
    switch (true) {
      case (((rectPts[0].lng - rectPts[2].lng) < 0) && ((rectPts[0].lat - rectPts[2].lat) > 0)):
        return 'lefttop2rightbottom';
      case (((rectPts[0].lng - rectPts[2].lng) < 0) && ((rectPts[0].lat - rectPts[2].lat) < 0)):
        return 'leftbottom2righttop';
      case (((rectPts[0].lng - rectPts[2].lng) > 0) && ((rectPts[0].lat - rectPts[2].lat) > 0)):
        return 'righttop2leftbottom';
      case (((rectPts[0].lng - rectPts[2].lng) > 0) && ((rectPts[0].lat - rectPts[2].lat) < 0)):
        return 'rightbottom2lefttop';
    }
  }

  /**
   * 根据类型，返回对应的左下角与右上角坐标
   * @param {number[][]} rectPts - 筛选框的四个角，顺序从起点开始顺时针排布
   * @param {string} type - 类型
   * @returns {number[][]} - [左下角坐标，右上角坐标]
   */
  function getLBandRT(rectPts, type) {
    switch (type) {
      case 'lefttop2rightbottom':
        return [rectPts[3], rectPts[1]];
      case 'leftbottom2righttop':
        return [rectPts[0], rectPts[2]];
      case 'righttop2leftbottom':
        return [rectPts[2], rectPts[0]];
      case 'rightbottom2lefttop':
        return [rectPts[1], rectPts[3]];
    }
  }



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
      // getPath 返回的边框角: 从起点开始顺时针排布
      const pts = e.overlay.getPath();
      // console.log(pts);
      // 判断筛选框绘制方向
      const directionType = paintDirection(pts);
      // 获取左下角与右上角
      const [leftBottom, rightTop] = getLBandRT(pts, directionType);
      const ptLeftBottom = new BMap.Point(leftBottom.lng, leftBottom.lat);
      const ptRightTop = new BMap.Point(rightTop.lng, rightTop.lat);
      // 创建 bound 需要边界的西南角和东北角
      // See https://mapopen-pub-jsapi.bj.bcebos.com/jsapi/reference/jsapi_reference.html#a1b2
      const bound = new BMap.Bounds(ptLeftBottom, ptRightTop);
      // 临时存储对象
      const obj = {}
      for (let [key, value] of Object.entries(data)) {
        // 设置默认值 []
        Reflect.set(obj, key, []);
        for (let { id, coord: [lng, lat] } of Object.values(value)) {
          let pt = new BMap.Point(lng, lat);
          if (BMapLib.GeoUtils.isPointInRect(pt, bound)) {
            // 存储对应索引，方便查找
            // 此处返回一个新数组，触发 Object.is() 浅比较
            obj[key] = [...obj[key], id];
          }
        }
        setSelected(prev => ({
          ...prev,
          [key]: prev.hasOwnProperty(key) ? [...prev[key], ...obj[key]] : obj[key]
        }))
      }
    })
  }, [map, data])


  useEffect(() => {
    // 向外暴露结果: 更像是外界通过一个方法从中取出结果
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