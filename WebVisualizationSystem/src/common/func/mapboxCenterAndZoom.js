import mapboxgl from 'mapbox-gl';

// mapbox 地图的自适应缩放及居中定位
/**
   * Pass the first coordinates in the LineString to `lngLatBounds` &
   * wrap each coordinate pair in `extend` to include them in the bounds
   * result. A variation of this technique could be applied to zooming
   * to the bounds of multiple Points or Polygon geomteries - it just
   * requires wrapping all the coordinates with the extend method.
   * 参考链接：https://blog.csdn.net/naipeng/article/details/53906652
   * @param {*} mapbox - mapbox地图ref对象
   * @param {*} coordinates - 地理坐标数组
   */
export function mapboxCenterAndZoom(mapbox, coordinates) {
  const map = mapbox.getMap();  // 获取地图实例

  // 以坐标0为初始值，边界逐渐扩展边界到最后一个坐标
  var bounds = coordinates.reduce(function (bounds, coord) {
    return bounds.extend(coord);   // extend(obj): 包含给定的经纬度或者经纬度边界来扩展区域边界
  }, new mapboxgl.LngLatBounds(coordinates[0], coordinates[0])); // new LngLatBounds(sw: [LngLatLike], ne: [LngLatLike]): 创建LngLatBounds的构造器，LngLatBounds对象表示一个地理上有界限的区域，使用西南和东北的点的经纬坐标表示

  // fitBounds(bounds,[options],[eventData])：移动缩放地图来将某个可视化区域包含在指定的地理边界内部，最终也会使用最高的zoomlevel来显示可视化区域试图
  map.fitBounds(bounds, {
    padding: 30
  })
}