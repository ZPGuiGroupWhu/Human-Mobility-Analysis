import { useEffect, useState } from 'react';
import { GPUGridLayer } from '@deck.gl/aggregation-layers';
import { PathLayer, ArcLayer } from '@deck.gl/layers';

export function useGetLayers(selectedTraj, histTrajs, visible) {
  const { spdShow, azmShow } = visible;

  // path-layer
  const [curPathLayer, setCurPathLayer] = useState(null);
  const [curArcLayer, setCurArcLayer] = useState(null);
  useEffect(() => {
    /**
     * Data format:
     * [
     *   {
     *     path: [[-122.4, 37.7], [-122.5, 37.8], [-122.6, 37.85]],
     *     name: 'Richmond - Millbrae',
     *     color: [255, 0, 0]
     *   },
     *   ...
     * ]
     */
    if (!selectedTraj) return () => { };
    const data = [{
      path: selectedTraj.data,
      name: selectedTraj.id,
      color: [59, 255, 245],
    }];
    setCurPathLayer(new PathLayer({
      id: 'cur-path-layer',
      data,
      pickable: true,
      widthScale: 20,
      widthMinPixels: 2,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: d => 5,
    }))
    const OD = [{
      from: {
        type: 'major',
        name: 'origin',
        coordinates: selectedTraj.data[0],
      },
      to: {
        type: 'major',
        name: 'destination',
        coordinates: selectedTraj.data.slice(-1)[0],
      }
    }]
    setCurArcLayer(new ArcLayer({
      id: 'cur-arc-layer',
      data: OD,
      visible: true,
      pickable: true,
      getStrokeWidth: 20,
      widthScale: 2,
      getSourcePosition: d => d.from.coordinates,
      getTargetPosition: d => d.to.coordinates,
      getSourceColor: [42, 255, 0],
      getTargetColor: [0, 143, 255]
    }))
  }, [selectedTraj]);

  // gpu-grid-layer-spd
  const [spdLayer, setSpdLayer] = useState(null);
  useEffect(() => {
    /**
     * Data format:
     * [
     *   {COORDINATES: [-122.42177834, 37.78346622], SPD: xxx},
     *   ...
     * ]
     */
    if (!selectedTraj) return () => { };
    let arr = [];
    for (let i = 0; i < selectedTraj.data.length; i++) {
      arr.push([...selectedTraj.data[i], selectedTraj.spd[i]]);
    }
    const data = arr.map(item => ({ COORDINATES: [item[0], item[1]], SPD: item[2] }));
    setSpdLayer(new GPUGridLayer({
      id: 'gpu-grid-layer-speed',
      data,
      visible: spdShow,
      pickable: true,
      extruded: true,//是否显示为3D效果
      cellSize: 50,//格网宽度，默认为100m
      elevationScale: 1,
      elevationAggregation: 'MAX',// 选用速度最大值作为权重
      colorAggregation: 'MAX',
      getPosition: d => d.COORDINATES,
      getElevationWeight: d => d.SPD,
      getColorWeight: d => d.SPD,
    }))
  }, [selectedTraj, spdShow]);


  // gpu-grid-layer-azimuth
  const [azmLayer, setAzmLayer] = useState(null);
  useEffect(() => {
    /**
     * Data format:
     * [
     *   {COORDINATES: [-122.42177834, 37.78346622], AZM: xxx},
     *   ...
     * ]
     */
    if (!selectedTraj) return () => { };
    let arr = [];
    for (let i = 0; i < selectedTraj.data.length; i++) {
      // azimuth 取 0 导致 grid 渲染有问题，目前解决方案为：统一为 azimuth 加上基底
      arr.push([...selectedTraj.data[i], (10 + selectedTraj.azimuth[i] * 10)]);
    }
    const data = arr.map(item => ({ COORDINATES: [item[0], item[1]], AZM: item[2] }));
    setAzmLayer(new GPUGridLayer({
      id: 'gpu-grid-layer-azimuth',
      data,
      visible: azmShow,
      pickable: true,
      extruded: true,//是否显示为3D效果
      cellSize: 50,//格网宽度，默认为100m
      elevationScale: 1,
      elevationAggregation: 'MAX',// 选用最大值作为权重
      colorAggregation: 'MAX',
      getPosition: d => d.COORDINATES,
      getElevationWeight: d => d.AZM,
      getColorWeight: d => d.AZM,
    }))
  }, [selectedTraj, azmShow]);


  // hist D sequences arc-layers
  const [histArcLayer, setHistArcLayer] = useState(null);
  useEffect(() => {
    /**
     * Data format:
     * [
     * {
     *   "from": {
     *     "type": "major",
     *     "name": "San Francisco Int'l",
     *     "abbrev": "SFO",
     *     "coordinates": [
     *       -122.38347034444931,
     *       37.61702508680534
     *     ]
     *   },
     *   "to": {
     *     "type": "major",
     *     "name": "Liverpool John Lennon",
     *     "abbrev": "LPL",
     *     "coordinates": [
     *       -2.858620657849378,
     *       53.3363751054422
     *     ]
     *   }
     *   ...
     * ]
     */
    const Ds = histTrajs.map(item => (item.destination));
    let data = []
    for (let i=0; i<Ds.length-1; i++) {
      data.push({
        from: {
          type: 'major',
          name: 'origin',
          coordinates: Ds[i],
        },
        to: {
          type: 'major',
          name: 'destination',
          coordinates: Ds[i + 1],
        }
      })
    }
    setHistArcLayer(new ArcLayer({
      id: 'hist-arc-layer',
      data,
      visible: true,
      pickable: true,
      getStrokeWidth: 20,
      widthScale: 2,
      getSourcePosition: d => d.from.coordinates,
      getTargetPosition: d => d.to.coordinates,
      getSourceColor: [255, 240, 0],
      getTargetColor: [255, 35, 0]
    }))
  }, [histTrajs])

  return {
    ids: [
      'cur-path-layer',
      'cur-arc-layer',
      'gpu-grid-layer-speed',
      'gpu-grid-layer-azimuth',
      'hist-arc-layer',
    ],
    layers: [
      curPathLayer,
      curArcLayer,
      spdLayer,
      azmLayer,
      histArcLayer,
    ]
  }
}
