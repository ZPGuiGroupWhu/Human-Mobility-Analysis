import React, { Component } from 'react';
import { Drawer, Button } from 'antd';
import './Drawer.css';
import { withMouse } from './withMouse';
import { LeftCircleTwoTone, RightCircleTwoTone, UpCircleTwoTone, DownCircleTwoTone } from '@ant-design/icons';

const gap = 40; // 按钮与内容间的空隙，默认 'px'
class MyDrawer extends Component {
  static defaultProps = {
    nodeCSSName: '.main',
    initVisible: false,
  }

  /**
   * props
   * @param {Boolean} initVisible - 内容初始状态 (可视 / 隐藏)  
   * @param {String} type - 侧边栏类型 (left / right / top / bottom)
   * @param {Number} width - 抽屉宽度
   * @param {Number} height - 抽屉高度
   * @param {String} nodeCSSName - Drawer 挂载节点的 css 选择器
   * @param {Function} render - 子组件渲染函数
   * 互斥属性：所有设置了id的抽屉间互斥，必须传递以下参数
   * @param {Nmuber} id - 抽屉id
   * @param {Number} curId - 当前激活的抽屉id
   * @param {Function} setCurId - 设置当前激活id
   */
  constructor(props) {
    super(props);
    this.state = {
      btnVisible: false, // 按钮是否可视
      drawerVisible: props.id ? (props.id === props.curId) : props.initVisible, // 内容是否可视，若传递id则忽略initVisible设置
    };
    this.check();
  }

  // 检查多个抽屉是否设置为互斥状态
  check = () => {
    const { id, curId, setCurId } = this.props;
    if (![id, curId, setCurId].every(item => !!item) || ![id, curId, setCurId].every(item => !item)) {
      return;
    } else {
      throw new Error('id, curId, setCurId 要一同设置')
    }
  }

  calMargin = (type) => {
    if (type === 'left' || type === 'right') {
      return this.props.width;
    } else if (type === 'top' || type === 'bottom') {
      return this.props.height;
    }
  }

  // 按钮可视
  setBtnVisible = (prevProps, prevState) => {
    if (prevState.drawerVisible === false) {
      if (prevProps.position[this.props.type] < gap && prevProps.position[this.props.type] > 0) {
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
        prevProps.position[this.props.type] < (this.calMargin(this.props.type) + gap) &&
        prevProps.position[this.props.type] > (this.calMargin(this.props.type))
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


  // 根据类型不同，设置不同的 padding 样式
  getPadding = (type) => {
    if (type === 'left' || type === 'right') {
      return '5px 0 5px 0';
    } else if (type === 'top' || type === 'bottom') {
      return '0px 0px';
    }
  }
  // 根据类型不同，设置不同的按钮 icon
  getButtonIcon = (type) => {
    switch (type) {
      case 'left':
        return {
          FoldButton: LeftCircleTwoTone,
          UnFoldButton: RightCircleTwoTone,
        }
      case 'right':
        return {
          FoldButton: RightCircleTwoTone,
          UnFoldButton: LeftCircleTwoTone,
        }
      case 'top':
        return {
          FoldButton: UpCircleTwoTone,
          UnFoldButton: DownCircleTwoTone,
        }
      case 'bottom':
        return {
          FoldButton: DownCircleTwoTone,
          UnFoldButton: UpCircleTwoTone,
        }
      default:
        throw new Error('None Type Error');
    }
  };
  // 按钮水平居中 / 垂直居中
  getButtonCentered = (type) => {
    if (type === 'left' || type === 'right') {
      return {
        top: '50%',
        transform: 'translateY(-50%)',
      };
    } else if (type === 'top' || type === 'bottom') {
      return {
        left: '50%',
        transform: 'translateX(-50%)',
      }
    }
  }

  // 切换抽屉可视状态
  switchDrawerVisible = () => {
    // 若传递id，则需要在点击时，更新当前激活id
    if (this.props.id) {
      this.props.setCurId(this.props.id);
    }
    this.setState(prev => ({
      drawerVisible: !prev.drawerVisible,
    }))
  }

  componentDidUpdate(prevProps, prevState) {
    this.setBtnVisible(prevProps, prevState);

    // 若设置了id，表明采用了Drawer互斥，当前激活id与本身不匹配时，隐藏。
    if (prevProps.curId !== this.props.curId) {
      if (this.props.id && this.props.id !== this.props.curId) {
        this.setState({
          drawerVisible: false,
        })
      }
    }
  }

  render() {
    const { FoldButton, UnFoldButton } = this.getButtonIcon(this.props.type); // 获取折叠/展开的图标
    return (
      <>
        <Drawer
          closable={false}
          width={this.props.width}
          height={this.props.height}
          keyboard
          mask={false}
          placement={this.props.type}
          visible={this.state.drawerVisible}
          bodyStyle={{
            padding: this.getPadding(this.props.type),
          }}
          getContainer={() => (document.querySelector(this.props.nodeCSSName))}
        >
          {this.props.render(this.state.drawerVisible)}
        </Drawer>
        <Button
          ghost
          shape="circle"
          icon={
            this.state.drawerVisible ?
              <FoldButton twoToneColor="#fff" /> :
              <UnFoldButton twoToneColor="#fff" />
          }
          style={{
            display: (this.state.btnVisible ? '' : 'none'),
            position: 'absolute',
            [this.props.type]: (this.state.drawerVisible ? this.calMargin(this.props.type) + 10 : 10) + 'px',
            ...(this.getButtonCentered(this.props.type)),
          }}
          onClick={() => { this.switchDrawerVisible() }}
        />
      </>
    )
  }
}

export default withMouse(MyDrawer);
