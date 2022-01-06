import { useEffect, useState } from 'react';
import { GPUGridLayer } from '@deck.gl/aggregation-layers';
import { PathLayer } from '@deck.gl/layers';

export function useGetLayers(selectedTraj, visible) {
  const { spdShow, azmShow } = visible;

  // path-layer
  const [pathLayer, setPathLayer] = useState(null);
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
    setPathLayer(new PathLayer({
      id: 'path-layer',
      data,
      pickable: true,
      widthScale: 20,
      widthMinPixels: 2,
      getPath: d => d.path,
      getColor: d => d.color,
      getWidth: d => 5,
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
      cellSize: 100,//格网宽度，默认为100m
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
      arr.push([...selectedTraj.data[i], selectedTraj.azimuth[i] || 1e-6]);
    }
    const data = arr.map(item => ({ COORDINATES: [item[0], item[1]], AZM: item[2] }));
    setAzmLayer(new GPUGridLayer({
      id: 'gpu-grid-layer-azimuth',
      data,
      visible: azmShow,
      pickable: true,
      extruded: true,//是否显示为3D效果
      cellSize: 200,//格网宽度，默认为100m
      elevationScale: 1,
      elevationAggregation: 'MAX',// 选用最大值作为权重
      colorAggregation: 'MAX',
      getPosition: d => d.COORDINATES,
      getElevationWeight: d => d.AZM,
      getColorWeight: d => d.AZM,
    }))
  }, [selectedTraj, azmShow]);

  return {
    ids: [
      'path-layer',
      'gpu-grid-layer-speed',
      'gpu-grid-layer-azimuth',
    ],
    layers: [
      pathLayer,
      spdLayer,
      azmLayer,
    ]
  }
}
