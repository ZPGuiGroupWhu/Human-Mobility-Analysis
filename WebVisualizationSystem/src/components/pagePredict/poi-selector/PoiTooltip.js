import React from 'react';
import { Card } from 'antd';

export default function PoiTooltip(props) {
  const {
    poiName,
    coord,
    url,
    address,
  } = props;

  return (
    <Card
      title={poiName}
      extra={<a href={url}>{'详情>>'}</a>}
      style={{ width: 300 }}
      bodyStyle={{overflow: 'auto'}}
      size='small'
      bordered={false}
      hoverable
    >
      <p>{`经纬度：${coord}`}</p>
      <p>{`详细地址：${address}`}</p>
    </Card>
  )
}