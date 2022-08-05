import { CPUGridLayer } from '@deck.gl/aggregation-layers';
import _ from 'lodash';


// 筛选 cell 单元格内占比最大的 type 类别 对应的 序号，返回颜色值
const getColorValueByPoiGridLayer = (points) => {
  const map = _.countBy(points, (item) => item.typeId);
  const idx = Object.values(map).findIndex(
    (item, idx, arr) => item === Math.max(...arr)
  );
  const value = poiTypes.findIndex((type) => {
    return +Object.keys(map)[idx] === type;
  });
  return value;
};

// 基于 cell 单元格内聚合的 POI 点数返回对应的 bar height
const getElevationValueByPoiGridLayer = (points) => points.length;


// PoiMap 三维柱状图
export const PoiMap = (data, visible, colorRange, flyToFocusPoint) =>{
  const layer = new CPUGridLayer({
    id: "poi-map",
    visible: visible,
    colorDomain: [0, 11],
    colorRange: colorRange,
    data: data,
    pickable: true,
    extruded: true,
    cellSize: 400,
    elevationScale: 4,
    getPosition: (d) => d.location,
    getColorValue: getColorValueByPoiGridLayer,
    getElevationValue: getElevationValueByPoiGridLayer,
    onClick: (info) => {
        flyToFocusPoint(info?.object?.position);
    }
  });
  return layer;
}
