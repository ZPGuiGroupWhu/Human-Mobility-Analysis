import BMap from 'BMap';

export default class SearchPOI {
  constructor(bmap,
    { panel = '', selectFirstResult = false, autoViewport = false } = {},
    { setSearchCompleteResult = undefined } = {}, // 回调函数
  ) {
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
    this.init({ setSearchCompleteResult });
  }

  // 实例初始化后，挂载一些回调函数
  init({ setSearchCompleteResult }) {
    // 挂载检索结束后的回调函数
    this.search.setSearchCompleteCallback((results) => {
      if (setSearchCompleteResult) {
        // 将检索结果转换格式后存储为 react state
        let obj = this.getSearchNums(results);
        setSearchCompleteResult(obj);
      }
    })
    // 清除标记
    // this.search.setMarkersSetCallback((pois) => {
    //   const vm = this;
    //   const markerNodes = document.getElementsByClassName("BMap_Marker");
    //   (Object.prototype.toString.call(markerNodes) === '[object HTMLCollection]') && Array.prototype.forEach.call(markerNodes, item => {
    //     item.style.setProperty('display', 'none');
    //     vm.bmap.removeOverlay();
    //   });
    // })
  }

  searchInCircle({ keyword, center, radius }) {
    const pt = new BMap.Point(...center);
    this.search.searchNearby(keyword, pt, radius);
  }

  /**
   * 获取搜索结果总数
   * @param {LocalResult | Array<LocalResult>} res - 检索结果
   * @returns - {value: number, name: string}[]
   */
  getSearchNums(res) {
    let obj;
    if (Array.isArray(res)) {
      obj = res?.reduce((prev, item) => {
        return [...prev, { value: item.getNumPois(), name: item.keyword }]
      }, [])
    } else {
      obj = Reflect.set({}, res.keyword, res.getNumPois());
    }
    return obj;
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
    this.circle = new BMap.Circle(pt, radius + 100, curOptions);
    this.bmap.addOverlay?.(this.circle);
  }

  addAndSearchInCircle({ keyword, center, radius }, circleStyle = {}) {
    this.searchInCircle({ keyword, center, radius });
    this.addCircleLayer(center, radius, circleStyle);
  }

  removeOverlay() {
    this.bmap.removeOverlay?.(this.circle);
    this.search.getResults() && this.search.clearResults();
  }
}