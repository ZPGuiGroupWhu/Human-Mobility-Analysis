import React, { Component } from 'react';
import { Drawer, Button } from 'antd';
import './LeftDrawer.css';
import { withMouse } from './withMouse';
import { LeftCircleTwoTone, RightCircleTwoTone } from '@ant-design/icons';

const leftBorder = 40;
class LeftDrawer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      btnVisible: false,
      drawerVisible: props.initVisible ?? false,
    };
  }

  setBtnVisible = (prevProps, prevState) => {
    if (prevState.drawerVisible === false) {
      if (prevProps.position.left < leftBorder && prevProps.position.left > 0) {
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
        prevProps.position.left < (prevProps.width + leftBorder) &&
        prevProps.position.left > (prevProps.width)
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
          width={this.props.width}
          keyboard
          mask={false}
          placement='left'
          visible={this.state.drawerVisible}
          bodyStyle={{
            padding: '5px 0 5px 0',
          }}
          getContainer={() => (document.querySelector('.main'))}
        >
          {this.props.render()}
        </Drawer>
        <Button
          ghost
          shape="circle"
          icon={
            this.state.drawerVisible ?
              <LeftCircleTwoTone twoToneColor="#fff" /> :
              <RightCircleTwoTone twoToneColor="#fff" />
          }
          style={{
            display: (this.state.btnVisible ? '' : 'none'),
            position: 'absolute',
            left: (this.state.drawerVisible ? this.props.width + 10 : 10) + 'px',
            top: '50%',
            transform: 'translateY(-50%)',
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

export default withMouse(LeftDrawer);
