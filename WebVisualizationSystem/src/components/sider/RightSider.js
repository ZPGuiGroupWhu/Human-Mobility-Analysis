import React, { useState, useRef, useEffect } from 'react';
import { Button } from 'antd';
import './sider.scss';
import './rightSider.scss'
import { createFromIconfontCN } from '@ant-design/icons';
import { MouseXY } from '@/components/layout/Layout'

const IconFont = createFromIconfontCN({
  scriptUrl: '//at.alicdn.com/t/font_2592318_lficp82yfh.js',
});


export default function RightSider(props) {
  const divRef = useRef(null);
  const [isShow, setShow] = useState(false);
  const curClass = isShow ? 'sider sider-right sider-right-active' : 'sider sider-right';

  let maxRight = window.innerWidth - 100, rec = 100;
  useEffect(()=>{
    // 需要在动画时间结束后，获取 div 最新的 right 值
    setTimeout(()=>{
      maxRight = isShow ? (divRef.current.getBoundingClientRect().left - rec) : window.innerWidth - rec
    }, 500)
  },[isShow])

  return (
    <MouseXY.Consumer>
      {mouseXY => {
        // console.log(mouseXY);
        return (
          <div className={curClass} ref={divRef}>
            <Button
              type="ghost"
              shape="circle"
              icon={!isShow ? <IconFont type='icon-leftarrowheads-copy' /> : <IconFont type='icon-rightarrowheads-copy' />}
              onClick={() => setShow(!isShow)}
              className='sider-btn sider-btn-right'
              style={{ display: (mouseXY[0] < maxRight) && 'none' }}
            />
            {props.children}
          </div>
        )
      }}
    </MouseXY.Consumer>
  )
}
