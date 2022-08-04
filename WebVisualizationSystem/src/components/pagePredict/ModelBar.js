import React from 'react';
import IconBtn from '@/components/IconBtn.js';
import {
  selectBlack,
  stopBlack,
  startBlack,
  settingBlack,
  cancelBlack,
} from '@/icon';


export default function ModelBar(props) {
  const {
    startPredict,
    stopPredict,
    clearPredict,
  } = props;

  return (
    <>
        {/* <IconBtn
          title='模型选择'
          imgSrc={selectBlack}
          clickCallback={null}
        />
        <IconBtn
          title='参数配置'
          imgSrc={settingBlack}
          clickCallback={null}
        /> */}
        <IconBtn
          title='开始预测'
          imgSrc={startBlack}
          clickCallback={startPredict}
        />
        <IconBtn
          title='暂停预测'
          imgSrc={stopBlack}
          clickCallback={stopPredict}
        />
        <IconBtn
          title='清除预测效果'
          imgSrc={cancelBlack}
          clickCallback={clearPredict}
        />
    </>
  )
}