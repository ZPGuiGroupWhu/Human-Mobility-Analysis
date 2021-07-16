import BMap from 'BMap';

export default class SearchPOI {
  constructor(bmap, { panel = '', selectFirstResult = false, autoViewport = false } = {}) {
    const opts = {
      renderOptions: {
        map: bmap,
        panel,
        selectFirstResult,
        autoViewport,
      }
    }
    this.bmap = bmap;
    this.circle = null; // 圆形覆盖物
    this.search = new BMap.LocalSearch(bmap, opts);
  }

  searchInCircle({keyword, center, radius}) {
    const pt = new BMap.Point(...center);
    this.search.searchNearby(keyword, pt, radius);
  }

  /**
   * 添加圆形覆盖物
   * @param {number[]} point - [lng, lat]
   * @param {number} radius
   * @param {object} opts
   */
  addCircleLayer(point, radius, opts = {}) {
    const defaultOptions = {
      strokeColor: '#00FBFF', //圆形边线颜色
      fillColr: '#00FBFF', // 圆形填充颜色。当参数为空时，圆形将没有填充效果
      strokeWeight: 2, // 圆形边线的宽度，以像素为单位
      strokeOpacity: 1, // 圆形边线透明度，取值范围0 - 1
      fillOpacity: .3, // 圆形填充的透明度，取值范围0 - 1
      strokeStyle: 'solid', // 圆形边线的样式，solid或dashed
    }
    let curOptions = Object.assign(defaultOptions, opts);

    const pt = new BMap.Point(...point);
    this.circle = new BMap.Circle(pt, radius, curOptions);
    this.bmap.addOverlay(this.circle);
  }

  addAndSearchInCircle({keyword, center, radius}, circleStyle={}) {
    this.searchInCircle({keyword, center, radius});
    this.addCircleLayer(center, radius, circleStyle);
  }

  removeOverlay() {
    this.bmap.removeOverlay(this.circle)
  }
}