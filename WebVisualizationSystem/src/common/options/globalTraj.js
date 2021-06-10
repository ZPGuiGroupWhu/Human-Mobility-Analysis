import { trajColorBar } from '@/common/options/trajColorBar';

const staticOptions = {
  type: "lines",
  coordinateSystem: "bmap",
  polyline: true,
  data: [],
  silent: true,
  lineStyle: {
    opacity: 0.2,
    width: 1,
  },
  progressiveThreshold: 200,
  progressive: 200,
}
const dynamicOptions = {
  type: "lines",
  coordinateSystem: "bmap",
  polyline: true,
  data: [],
  lineStyle: {
    width: 0,
  },
  effect: {
    constantSpeed: 20,
    show: true,
    trailLength: 0.2,
    symbolSize: 1.5,
  },
  zlevel: 1,
}

const staticNames = trajColorBar.map(item => item.static);
const dynamicNames = trajColorBar.map(item => item.dynamic);

export const globalStaticTraj = staticNames.map((name, idx) => ({ name, ...staticOptions, zlevel: 90 + idx }));
export const globalDynamicTraj = dynamicNames.map((name, idx) => ({ name, ...dynamicOptions, zlevel: idx + 1 }));