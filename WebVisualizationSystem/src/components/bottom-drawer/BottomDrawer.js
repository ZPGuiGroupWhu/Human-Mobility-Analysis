import React, { Component } from 'react';
import { Drawer, Button } from 'antd';
import './BottomDrawer.css';
import { withMouse } from './withMouse';
import { UpCircleTwoTone, DownCircleTwoTone } from '@ant-design/icons';

const bottomBorder = 40;
class BottomDrawer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      btnVisible: false,
      drawerVisible: false,
    };
  }

  setBtnVisible = (prevProps, prevState) => {
    if (prevState.drawerVisible === false) {
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
    if (prevState.drawerVisible === true) {
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
          bodyStyle={{
            padding: '0 100px',
          }}
        >
          {this.props.render()}
        </Drawer>
        <Button
          ghost
          shape="circle"
          disabled={this.props.bottomBtnDisabled}
          icon={
            this.state.drawerVisible ?
              <DownCircleTwoTone twoToneColor="#fff" /> :
              <UpCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            display: (this.state.btnVisible ? '' : 'none'),
            position: 'absolute',
            bottom: (this.state.drawerVisible ? this.props.height + 10 : 10) + 'px',
            left: '50%',
            transform: 'translateX(-50%)',
          }}
          onClick={(e) => {
            this.setState(prev => ({
              drawerVisible: !prev.drawerVisible,
            }))
          }}
        />
      </>

    )
  }
}

export default withMouse(BottomDrawer);
