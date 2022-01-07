import React, { useEffect, useRef, useState, useCallback, useMemo, useLayoutEffect, useReducer } from 'react';
import DeckGL from '@deck.gl/react';
import { StaticMap } from 'react-map-gl';
import { MapboxLayer } from '@deck.gl/mapbox';
import { Radio } from 'antd';
// 函数
import { mapboxCenterAndZoom } from '@/common/func/mapboxCenterAndZoom';
import eventBus, { HISTACTION } from '@/app/eventBus';
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

  const [radioHeight, setRadioHeight] = useState(0);
  useLayoutEffect(() => {
    const radioObj = document.querySelector('div.ant-radio-group.ant-radio-group-solid.ant-radio-group-small');
    setRadioHeight(radioObj.offsetHeight);
  }, [])

  // 坐标改变，mapbox 自适应 center & zoom
  useEffect(() => {
    if (mapRef.current && coords.length) {
      mapboxCenterAndZoom(mapRef.current, coords)
    }
  }, [coords])


  // 图层可视状态管理
  const initGridLayerVisible = {
    spdShow: false,
    azmShow: false,
  }
  const gridLayerVisibleReducer = (state, action) => {
    switch (action.type) {
      case 'spd':
        return {
          ...initGridLayerVisible,
          spdShow: true,
        }
      case 'azm':
        return {
          ...initGridLayerVisible,
          azmShow: true,
        }
      case 'none':
        return { ...initGridLayerVisible }
      default:
        break;
    }
  }
  const [gridLayerVisible, gridLayerVisibleDispatch] = useReducer(gridLayerVisibleReducer, initGridLayerVisible)
  const onGridLayerChange = (e) => {
    switch (e.target.value) {
      case 'spd':
        gridLayerVisibleDispatch({ type: 'spd' });
        break;
      case 'azm':
        gridLayerVisibleDispatch({ type: 'azm' });
        break;
      case 'none':
        gridLayerVisibleDispatch({ type: 'none' });
        break;
      default:
        break;
    }
  }

  const [histTrajs, setHistTrajs] = useState([]); // 存放历史轨迹数据
  useEffect(() => {
    // 获取前N天历史轨迹数据：数据组织+坐标纠偏
    eventBus.on(HISTACTION, (histTrajs) => { setHistTrajs(histTrajs) });
  }, [])

  // ids & deckgl-layers
  const { ids, layers } = useGetLayers(selectedTraj, histTrajs, gridLayerVisible);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Radio.Group size={"small"} style={{ width: '100%', display: 'flex' }} buttonStyle="solid" onChange={onGridLayerChange} defaultValue="none">
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="spd">速度</Radio.Button>
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="azm" >转向角</Radio.Button>
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="none">关闭</Radio.Button>
      </Radio.Group>
      {/* deckgl-react-mapbox: https://deck.gl/docs/api-reference/mapbox/overview#using-with-react */}
      <section style={{ position: 'absolute', inset: `${radioHeight}px 0 0 0` }}>
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
          layers={layers}>
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
          }}>{selectedTraj.id}</span>
      </section>
    </div>
  )
}
