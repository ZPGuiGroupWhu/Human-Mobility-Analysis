import React, { Component } from 'react';
import { Drawer, Button } from 'antd';
import './CalendarDrawer.css';
import { withMouse } from './withMouse';
import { UpCircleTwoTone, DownCircleTwoTone } from '@ant-design/icons';

const bottomBorder = 40;
class CalendarDrawer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      btnVisible: false,
      // drawerVisible: false,
    };
  }

  setBtnVisible = (prevProps, prevState) => {
    if (prevProps.bottomDrawerVisible === false) {
      if (prevProps.position.bottom < bottomBorder && prevProps.position.bottom > 0) {
        !prevState.btnVisible && this.setState({
          btnVisible: true,
        })
      } else {
        prevState.btnVisible && this.setState({
          btnVisible: false,
        })
      }
    }
    if (prevProps.bottomDrawerVisible === true) {
      if (
          prevProps.position.bottom < (prevProps.height + bottomBorder) &&
          prevProps.position.bottom > (prevProps.height)
      ) {
        !prevState.btnVisible && this.setState({
          btnVisible: true
        })
      } else {
        prevState.btnVisible && this.setState({
          btnVisible: false
        })
      }
    }
  };

  //将按钮状态返回给父组件
  toParent = (drawer) => {
    this.props.setDrawerState(drawer);
  };

  componentDidUpdate(prevProps, prevState) {
    this.setBtnVisible(prevProps, prevState);
  }

  render() {
    return (
        <>
          <Drawer
              closable={false}
              height={this.props.height}
              keyboard
              mask={false}
              placement='bottom'
              visible={this.props.bottomDrawerVisible}
              bodyStyle={{
                padding: '0 100px',
              }}
          >
            {this.props.render()}
          </Drawer>
          <Button
              ghost
              shape="circle"
              icon={
                this.props.bottomDrawerVisible ?
                    <DownCircleTwoTone twoToneColor="#fff" /> :
                    <UpCircleTwoTone twoToneColor="#fff" />
              }
              style={{
                display: (this.state.btnVisible ? '' : 'none'),
                position: 'absolute',
                bottom: (this.props.bottomDrawerVisible ? this.props.height + 10 : 10) + 'px',
                left: '50%',
                transform: 'translateX(-50%)',
              }}
              onClick={(e) => {
                this.toParent('bottom');
              }}
          />
        </>

    )
  }
}

export default withMouse(CalendarDrawer);