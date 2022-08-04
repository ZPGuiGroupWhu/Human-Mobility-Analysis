import React, { useEffect, useRef, useState, useCallback, useMemo, useLayoutEffect, useReducer } from 'react';
import DeckGL from '@deck.gl/react';
import { StaticMap } from 'react-map-gl';
import { MapboxLayer } from '@deck.gl/mapbox';
import { Radio } from 'antd';
// 函数
import { mapboxCenterAndZoom } from '@/common/func/mapboxCenterAndZoom';
import eventBus, { HISTACTION, SPDAZMACTION } from '@/app/eventBus';
// Hooks
import { useSingleTraj } from '@/project/predict/function/useSingleTraj';
import { useGetLayers } from './useGetLayers';


// MAPBOX 密钥
const MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiMjAxNzMwMjU5MDE1NyIsImEiOiJja3FqM3RjYmIxcjdyMnhsbmR0bHo2ZGVpIn0.wNBmzyxhzCMx9PhIH3rwCA';

export default function Map3D(props) {
  const [glContext, setGLContext] = useState();

  const deckRef = useRef(null); // deckgl ref
  const mapRef = useRef(null); // mapbox ref
  const onMapLoad = useCallback((ids, coordinates) => {
    const map = mapRef.current.getMap();
    const deck = deckRef.current.deck;
    // You must initialize an empty deck.gl layer to prevent flashing
    ids.forEach(id => {
      // This id has to match the id of the deck.gl layer
      map.addLayer(new MapboxLayer({ id, deck }))
    })
    mapboxCenterAndZoom(mapRef.current, coordinates)
  }, []);

  const selectedTraj = useSingleTraj(); // 从候选列表中选取一条轨迹(用于展示)
  const coords = useMemo(() => (selectedTraj ? selectedTraj.data : []), [selectedTraj]); // 轨迹坐标数组

  // 坐标改变，mapbox 自适应 center & zoom
  useEffect(() => {
    if (mapRef.current && coords.length) {
      mapboxCenterAndZoom(mapRef.current, coords)
    }
  }, [coords])


  const [histTrajs, setHistTrajs] = useState([]); // 存放历史轨迹数据
  const [gridLayerVisible, setGridLayerVisible] = useState({}); // 速度&转向角图层是否可视
  useEffect(() => {
    // 获取前N天历史轨迹数据：数据组织+坐标纠偏
    eventBus.on(HISTACTION, (histTrajs) => { setHistTrajs(histTrajs) });
    // 速度/转向角按钮事件
    eventBus.on(SPDAZMACTION, (val) => { setGridLayerVisible(val) });
  }, [])



  // ids & deckgl-layers
  const { ids, layers, tooltipInfo } = useGetLayers(selectedTraj, histTrajs, gridLayerVisible);

  return (
    <div style={{ width: '100%', position: 'relative', paddingTop: '100%' }}>
      {/* deckgl-react-mapbox: https://deck.gl/docs/api-reference/mapbox/overview#using-with-react */}
      <section style={{ position: 'absolute', inset: '0' }}>
        <DeckGL
          ref={deckRef}
          initialViewState={{
            longitude: 114.17,
            latitude: 22.65,
            zoom: 10,
            pitch: 45,
            bearing: 0
          }}
          controller={true}
          onWebGLInitialized={setGLContext}
          glOptions={{
            stencil: true
          }}
          layers={layers}
        >
          {glContext && (
            <StaticMap
              ref={mapRef}
              gl={glContext}
              onLoad={() => { onMapLoad(ids, coords) }}
              mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN}
              mapStyle={'mapbox://styles/2017302590157/cksbi52rm50pk17npkgfxiwni'}
            />
          )}
        </DeckGL>
        {/* 当前轨迹标签 */}
        <span
          style={{
            position: 'absolute',
            top: '5px',
            left: '5px',
            color: '#fff',
            fontWeight: 'bold'
          }}>{selectedTraj?.id}
        </span>
        <div style={{
          position: 'absolute',
          zIndex: 999,
          pointerEvents: 'none',
          left: tooltipInfo.x,
          top: tooltipInfo.y,
          color: '#fff',
          backgroundColor: 'rgba(100,100,100,0.5)', "whiteSpace": "pre"
        }}
        >
          {!tooltipInfo.data ? tooltipInfo.data : (`历史数据集第${tooltipInfo.idx}条轨迹\n出发日期: ${tooltipInfo.data.from.time.split(' ')[0]}\n出发时刻:${tooltipInfo.data.from.time.split(' ')[1]}\n到达时刻:${tooltipInfo.data.to.time.split(' ')[1]}`)}
        </div>
      </section>
    </div >
  )
}
