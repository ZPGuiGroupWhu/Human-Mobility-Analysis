import React, { useEffect, useReducer, useState } from 'react';
import BMap from 'BMap';

export default function ScatterTolltip(props) {
  const { title, lng, lat } = props;

  // 逆地址解析
  const geocoder = (lng, lat) => {
    // 创建地理编码实例, 并配置参数获取乡镇级数据
    let myGeo = new BMap.Geocoder({ extensions_town: true });
    // 根据坐标得到地址描述
    return new Promise((resolve, reject) => {
      myGeo.getLocation(new BMap.Point(lng, lat), function (result) {
        if (result) {
          resolve({
            address: result.address,
            poi: result.surroundingPois[0]?.title ?? '无',
          })
        }
      })
    })
  }

  const initialState = {
    address: '解析中...',
    poi: '解析中...',
  }
  function geoReducer(state, action) {
    switch (action.type) {
      case 'address':
        return {
          ...state,
          address: action.payload,
        }
      case 'poi':
        return {
          ...state,
          poi: action.payload,
        }
      default:
        return {
          ...state,
        }
    }
  }
  const [geoResult, geoResultDispatch] = useReducer(geoReducer, initialState);
  useEffect(() => {
    async function getResult() {
      let obj = await geocoder(lng, lat);
      geoResultDispatch({ type: 'address', payload: obj.address });
      geoResultDispatch({ type: 'poi', payload: obj.poi });
    }
    getResult();
  }, [lng, lat])


  return (
    <>
      <div style={{ color: '#fff' }}>{title}</div>
      <div style={{ color: '#fff' }}><strong>经度:</strong> {lng}</div>
      <div style={{ color: '#fff' }}><strong>纬度:</strong> {lat}</div>
      <div style={{ color: '#fff' }}><strong>地址:</strong> {geoResult.address}</div>
      <div style={{ color: '#fff' }}><strong>POI:</strong> {geoResult.poi}</div>
      {props.children}
    </>
  )
}

// orgInfo.value?.[1].toFixed(3)