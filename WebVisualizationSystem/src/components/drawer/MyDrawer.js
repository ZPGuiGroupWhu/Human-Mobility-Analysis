import React from 'react'
import BottomDrawer from './BottomDrawer';
import LeftDrawer from './LeftDrawer';
import RightDrawer from './RightDrawer';

export default function MyDrawer(props) {
  const { mode, modeStyle } = props;

  const choice = {
    left: () => (
    <LeftDrawer
      {...modeStyle}
    >
      {props.children}
    </LeftDrawer>
    ),
    right: () => (
      <RightDrawer
        {...modeStyle}
      >
        {props.children}
      </RightDrawer>
    ),
    bottom: () => (
      <BottomDrawer
        {...modeStyle}
      >
        {props.children}
      </BottomDrawer>
    ),
  }

  return choice[mode]()
}