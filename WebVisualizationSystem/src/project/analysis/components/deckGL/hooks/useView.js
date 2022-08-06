import { useState } from "react";
import { INITIAL_VIEW_STATE } from "../configs/poiMap/Config";

/**
 * 地图视角操作逻辑
 */
export const useView = (initViewState = INITIAL_VIEW_STATE) => {
  // 上一(初始)帧
  const [prevViewState, setViewState] = useState(initViewState);
  // 视角切换过渡效果
  const flyToFocusPoint = (location) => {
    setViewState((state) => ({
      ...state,
      longitude: location[0],
      latitude: location[1],
      zoom: initViewState.zoom + 3,
    }));
  };

  return {
    prevViewState,
    setViewState,
    flyToFocusPoint,
  };
};
