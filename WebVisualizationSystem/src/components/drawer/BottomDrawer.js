import React, { useState, useEffect, useRef } from 'react';
import { Button } from 'antd';
import { UpOutlined, DownOutlined } from '@ant-design/icons';
import './BottomDrawer.scss';
import { eventEmitter } from '@/common/func/EventEmitter';
import { createFromIconfontCN } from '@ant-design/icons';

const IconFont = createFromIconfontCN({
  scriptUrl: '//at.alicdn.com/t/font_2648021_r6kr8s162g.js',
});


export default function BottomDrawer(props) {
  const {
    // 建议自定义更改
    boxHeight = '200px', // 组件整体高度
    left = 0, // 相对偏移(左)
    right = 0, // 相对偏移(右)
    btnOpenForbidden = false, // 是否禁用展开按钮

    // 无需更改
    buttonWidth = '30px', // 按钮宽度 + 空白间距
    showBottom = '5px', // 显示时组件距离底部的距离
    buttonPadding = '10px', // 按钮与主体内容间的距离
  } = props;

  const boxStyle = {
    left,
    right,
    height: boxHeight,
  }

  const [isShow, setShow] = useState(false); // 主体显示
  const [btnShow, setBtnShow] = useState(false); // 按钮显示

  const document = useRef(null);

  useEffect(() => {
    const doc = document.current;
    if (!doc) return () => { };

    const mouseover = () => { (btnOpenForbidden) || setBtnShow(true); }
    const mouseout = () => { setBtnShow(false) }

    doc.addEventListener('mouseover', mouseover);
    doc.addEventListener('mouseout', mouseout);

    return () => {
      doc.removeEventListener('mouseover', mouseover);
      doc.removeEventListener('mouseout', mouseout);
    }
  }, [document, isShow])

  useEffect(() => {
    eventEmitter.on('showCalendar', () => { setShow(prev => !prev) });
  }, [])

  return (
    <div
      style={{
        ...boxStyle,
        bottom: isShow ? showBottom : `-${parseInt(boxHeight) - parseInt(buttonWidth)}px`
      }}
      className='bottom-drawer'
      ref={document}
    >
      <div style={{ width: buttonWidth, paddingBottom: buttonPadding }}>
        <Button
          block
          size='small'
          type='ghost'
          shape='circle'
          icon={isShow ? <IconFont type='icon-down-arrow-copy' /> : <IconFont type='icon-up-arrow-copy' />}
          onClick={(e) => { setShow(prev => !prev); setBtnShow(false) }}
          style={{ visibility: btnShow ? 'visible' : 'hidden' }}
        ></Button>
      </div>
      <div
        className='main'
      >
        {props.children}
      </div>
    </div>
  )
}
