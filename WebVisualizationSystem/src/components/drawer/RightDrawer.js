import React, { useState, useEffect, useRef } from 'react';
import { Button, Space } from 'antd';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import './RightDrawer.scss';
import { eventEmitter } from '@/common/func/EventEmitter';

export default function RightDrawer(props) {
  const {
    // 建议自定义更改
    boxWidth = '80px', // 组件整体宽度
    top = 0, // 相对偏移(上)
    bottom = 0, // 相对偏移(下)
    backgroundColor = 'rgba(255, 255, 255, 1)',

    space = 40, // 内容间距
    btnOpenForbidden = false, // 是否禁用展开按钮
    borderRadius = '0 0 5 5',
    unfoldEventName = 'showRightDrawer',

    // 无需更改
    buttonWidth = '30px', // 按钮宽度 + 空白间距
    showRight = '5px', // 显示时组件距离底部的距离
    buttonPadding = '10px', // 按钮与主体内容间的距离
  } = props;

  const boxStyle = {
    top,
    bottom,
    width: boxWidth,
  }

  const [isShow, setShow] = useState(false); // 主体显示
  const [btnShow, setBtnShow] = useState(false); // 按钮显示

  const document = useRef(null);

  useEffect(() => {
    const doc = document.current;
    const mouseoverEvent = () => { (btnOpenForbidden && !isShow) || setBtnShow(true); };
    const mouseoutEvent = () => {setBtnShow(false)}
    if (!doc) return () => { };
    doc.addEventListener('mouseover', mouseoverEvent)
    doc.addEventListener('mouseout', mouseoutEvent)

    return () => {
      doc.removeEventListener('mouseover', mouseoverEvent);
      doc.removeEventListener('mouseout', mouseoutEvent);
    }
  }, [document, isShow])

  // 向外暴露控制组件展开的方法
  useEffect(() => {
    eventEmitter.on(unfoldEventName, () => { setShow(true) });
  }, [])

  return (
    <div
      style={{
        ...boxStyle,
        right: isShow ? showRight : `-${parseInt(boxWidth) - parseInt(buttonWidth)}px`
      }}
      className='right-drawer'
      ref={document}
    >
      <div style={{ width: buttonWidth, paddingRight: buttonPadding }}>
        <Button
          block
          size='small'
          type='ghost'
          shape='circle'
          icon={isShow ? <RightOutlined /> : <LeftOutlined />}
          onClick={(e) => { setShow(prev => !prev); setBtnShow(false) }}
          style={{ visibility: btnShow ? 'visible' : 'hidden' }}
        ></Button>
      </div>
      <div
        className='main'
        style={{
          backgroundColor,
          borderRadius,
        }}
      >
        <Space direction='vertical' size={space}>
          {React.Children.map(props.children, (child) => {
            return React.cloneElement(child, { isShow })
          })}
        </Space>
      </div>
    </div>
  )
}
