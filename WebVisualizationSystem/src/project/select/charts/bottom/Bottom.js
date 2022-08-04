import React from 'react';
import Drawer from '@/components/drawer/Drawer';
import ChartBottom from './ChartBottom';


export default function Bottom(props) {
  return (
    <Drawer 
      render={()=>(props.bottomHeight ? <ChartBottom {...props} /> : null)}
      width={props.bottomWidth}
      height={props.bottomHeight + 21}
      initVisible={true}
      nodeCSSName='.bottom'
      type="bottom"
    />
  )
}
