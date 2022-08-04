import React from 'react';
import IconBtn from '@/components/IconBtn.js';
import {
  odPoints,
  heatmap,
} from '@/icon';


export default function LegendBar(props) {
  const {fnList} = props;

  return (
    <>
        <IconBtn
          title='OD点'
          imgSrc={odPoints}
          clickCallback={fnList[0]}
        />
        <IconBtn
          title='OD热力图'
          imgSrc={heatmap}
          clickCallback={fnList[1]}
        />
    </>
  )
}