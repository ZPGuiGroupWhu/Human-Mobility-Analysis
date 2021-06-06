import React from 'react';
import LeftSider from './LeftSider'
import RightSider from './RightSider';

/**
 * @param {string} floatType - 浮动类型：'left' 'right'
 */
export default function Sider(props) {
  if (props.floatType === 'left') {
    return <LeftSider>{props.children}</LeftSider>
  } else if (props.floatType === 'right') {
    return <RightSider>{props.children}</RightSider>
  }
}
