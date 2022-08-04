import { useState, useReducer } from 'react';

export const usePoi = () => {
  // 控制 poi 查询功能是否开启
  const [poiDisabled, setPoiDisabled] = useState(false);

  function poiReducer(state, action) {
    const { type, payload } = action;

    switch (type) {
      // 关键词
      case 'keyword':
        return {
          ...state,
          keyword: payload
        }
      // 坐标点
      case 'center':
        return {
          ...state,
          center: payload
        }
      // 控制缓冲区半径  
      case 'radius':
        return {
          ...state,
          radius: payload
        }
      case 'description':
        return {
          ...state,
          description: payload
        }
      default:
        return;
    }
  }

  const initPoiState = {
    // 目标POI的搜索关键词
    keyword: '',
    // 缓冲区中心坐标
    center: null,
    // 缓冲区半径
    radius: 100,
    // 类型：起点/当前点/终点
    description: 'end'
  }

  const [poiState, poiDispatch] = useReducer(poiReducer, initPoiState);


  return {
    poiDisabled,
    setPoiDisabled,
    poiState,
    poiDispatch,
  }
}