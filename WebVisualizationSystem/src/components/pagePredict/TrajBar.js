import React from 'react';
import IconBtn from '@/components/IconBtn.js';
import {
  trajBlack,
  clearBlack,
} from '@/icon';


export default function TrajBar(props) {
  const {showAllTraj, clearTraj} = props;

  return (
    <>
        <IconBtn
          title='显示(所有)轨迹'
          imgSrc={trajBlack}
          clickCallback={showAllTraj}
        />
        <IconBtn
          title='清除(所有)轨迹'
          imgSrc={clearBlack}
          clickCallback={clearTraj}
        />
    </>
  )
}