import React from 'react';
import Chart from './ChartLeft';
import LeftDrawer from '@/components/left-drawer/LeftDrawer'; // 左侧抽屉


export default function Left(props) {

  return (
    <>
      <LeftDrawer 
        render={() => <Chart width={props.width} />}
        width={props.width}
        initVisible={true}
      />
    </>
  )
}
