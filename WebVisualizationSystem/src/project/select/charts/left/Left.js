import React from 'react';
import ChartLeft from './ChartLeft';
import Drawer from '@/components/drawer/Drawer'; // 左侧抽屉


export default function Left(props) {
  return (
    <>
      <Drawer 
        render={() => <ChartLeft {...props} />}
        width={props.width}
        initVisible={true}
        type='left'
      />
    </>
  )
}
