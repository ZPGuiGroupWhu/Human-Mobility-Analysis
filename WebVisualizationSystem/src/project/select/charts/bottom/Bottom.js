import React from 'react';
import Drawer from '@/components/drawer/Drawer';
import ChartBottom from './ChartBottom';


export default function Bottom(props) {
  return (
    <Drawer 
      render={()=>(props.bottomHeight ? <ChartBottom bottomHeight={props.bottomHeight} bottomWidth={props.bottomWidth} /> : null)}
      width={props.bottomWidth}
      height={props.bottomHeight}
      initVisible={true}
      nodeCSSName='.bottom'
      type="bottom"
    />
  )
}
