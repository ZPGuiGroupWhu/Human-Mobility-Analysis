import React from 'react';

export default function ScatterTolltip(props) {
  return (
    <>
      <div style={{ color: '#fff' }}>{props.title}</div>
      <div
        dangerouslySetInnerHTML={{ __html: `<strong>经度:</strong> ${props.lng}` }}
        style={{ color: '#fff' }}
      ></div>
      <div
        dangerouslySetInnerHTML={{ __html: `<strong>纬度:</strong> ${props.lat}` }}
        style={{ color: '#fff' }}
      ></div>
    </>
  )
}

// orgInfo.value?.[1].toFixed(3)