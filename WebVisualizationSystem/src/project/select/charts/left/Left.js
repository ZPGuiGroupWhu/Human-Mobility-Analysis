import React from 'react';
import Chart from './ChartLeft';
import Drawer from '@/components/drawer/Drawer'; // 左侧抽屉


export default function Left(props) {

  return (
    <>
      <Drawer 
        render={() => <Chart width={props.width} />}
        margin={props.width}
        initVisible={true}
        type='left'
      />
    </>
  )
}
