import BMap from 'BMap';
import BMapLib from 'BMapLib';

// See 
// 1. https://lbsyun.baidu.com/index.php?title=jspopular3.0/guide/mouse2draw
// 2. https://www.cnblogs.com/zdd2017/p/13495908.html

/**
 * @param {object} bmap - 百度地图实例
 */
class BMapBrush {
  constructor(bmap) {
    this.map = bmap;
    this.coords = {}; // 地图中的点数组 number[][]

    this.overLayers = []; // 覆盖物数组
    this.drawingManager = null;
    this.selected = {}; // 被选择的点

    this.init();
  }

  init() {
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
    this.drawingManager = new BMapLib.DrawingManager(this.map, {
      isOpen: false, //是否开启绘制模式
      enableDrawingTool: false, //是否显示工具栏
      circleOptions: styleOptions, //圆的样式
      polylineOptions: styleOptions, //线的样式
      polygonOptions: styleOptions, //多边形的样式
      rectangleOptions: styleOptions //矩形的样式
    });

    this.drawingManager.addEventListener('overlaycomplete', (e) => {
      // 记录覆盖物
      this.overLayers.push(e.overlay);
      // 获取选框内点信息
      // See https://www.cnblogs.com/zdd2017/p/13495908.html
      const pts = e.overlay.getPath();
      const leftTop = pts[3];
      const rightBottom = pts[1];
      const ptLeftTop = new BMap.Point(leftTop.lng, leftTop.lat);
      const ptRightBottom = new BMap.Point(rightBottom.lng, rightBottom.lat);
      const bound = new BMap.Bounds(ptLeftTop, ptRightBottom);
      for (let [key, value] of Object.entries(this.coords)) {
        for (let [idx, [lng, lat]] of Object.entries(value)) {
          let pt = new BMap.Point(lng, lat);
          if (BMapLib.GeoUtils.isPointInRect(pt, bound)) {
            // 存储对应索引，方便查找
            // 此处返回一个新数组，触发 Object.is() 浅比较
            this.selected[key] = [...this.selected[key], idx];
            this.selected = {...this.selected}
          }
        }
      }
      console.log(this.selected);
    })
  }

  // 清除覆盖物
  clearOverLayers() {
    for (let i = 0; i < this.overLayers.length; i++) {
      this.map.removeOverlay(this.overLayers[i]);
    }
    this.overLayers.length = 0
  }

  /**
   * 获取坐标数据
   * @param {string} key - 键名
   * @param {number[][]} value - 坐标数组
   * @param {number[][]} coords - 存储数组
   */
  getCoords(key, value) {
    this.coords[key] = value;
  }
}

export default BMapBrush;