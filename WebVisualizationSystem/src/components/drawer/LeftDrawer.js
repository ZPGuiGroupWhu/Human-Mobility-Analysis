import React, { useState, useEffect, useRef } from 'react';
import { Button, Space } from 'antd';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import './LeftDrawer.scss';
import { createFromIconfontCN } from '@ant-design/icons';

const IconFont = createFromIconfontCN({
  scriptUrl: '//at.alicdn.com/t/font_2648021_r6kr8s162g.js',
});

export default function LeftDrawer(props) {
  const {
    // 弹出位置
    mode = 'left',

    // 建议自定义更改
    boxWidth = '80px', // 组件整体高度
    top = 0, // 相对偏移(上)
    bottom = 0, // 相对偏移(下)
    backgroundColor = 'rgba(255, 255, 255, 1)',

    // 无需更改
    buttonWidth = '30px', // 按钮宽度 + 空白间距
    showLeft = '5px', // 显示时组件距离底部的距离
    buttonPadding = '10px', // 按钮与主体内容间的距离
  } = props;

  const boxStyle = {
    top,
    bottom,
    width: boxWidth,
  }

  const [isShow, setShow] = useState(true); // 主体显示
  const [btnShow, setBtnShow] = useState(false); // 按钮显示

  const document = useRef(null);

  useEffect(() => {
    const doc = document.current;
    if (!doc) return () => { };
    doc.addEventListener('mouseover', () => { setBtnShow(true); })
    doc.addEventListener('mouseout', () => setBtnShow(false))
  }, [document])

  return (
    <div
      style={{
        ...boxStyle,
        left: isShow ? showLeft : `-${parseInt(boxWidth) - parseInt(buttonWidth)}px`
      }}
      className='left-drawer'
      ref={document}
    >
      <div
        className='main'
        style={{
          backgroundColor,
        }}
      >
        <Space direction='vertical' size={40}>
          {React.Children.map(props.children, (child) => {
            return React.cloneElement(child, { isShow })
          })}
        </Space>
      </div>
      <div style={{ width: buttonWidth, paddingLeft: buttonPadding }}>
        <Button
          block
          size='small'
          type='ghost'
          shape='circle'
          icon={isShow ? <IconFont type='icon-left-arrow-copy' /> : <IconFont type='icon-right-arrow-copy' />}
          onClick={(e) => { setShow(prev => !prev); setBtnShow(false) }}
          style={{ visibility: btnShow ? 'visible' : 'hidden' }}
        ></Button>
      </div>
    </div>
  )
}
