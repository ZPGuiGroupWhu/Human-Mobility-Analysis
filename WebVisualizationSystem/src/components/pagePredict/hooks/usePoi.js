import React, { useState } from 'react';

export const usePoi = () => {
  // 控制 poi 查询功能是否开启
  const [poiDisabled, setPoiDisabled] = useState(false);

  // 控制缓冲区半径
  const [bufferValue, setBufferValue] = useState(30);


  return {
    poiDisabled,
    setPoiDisabled,
    bufferValue,
    setBufferValue,
  }
}