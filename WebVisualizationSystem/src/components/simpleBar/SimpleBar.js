import expStyle from './SimpleBar.scss';
import React, { PureComponent } from 'react';
import { createFromIconfontCN } from '@ant-design/icons';
import { Tooltip } from 'antd';

export default class SimpleBar extends PureComponent {
  timer = null;

  constructor(props) {
    super(props);
    this.state = {
      timer: null, // 计时器
      isUnfold: false, // 是否展开
      tooltipVisible: false, // 文本提示框显示状态
    }
    // ref 对象
    this.mainRef = React.createRef();
    // iconfont 图标
    this.IconFont = createFromIconfontCN({
      scriptUrl: this.props.iconScriptUrl,
    });
    // 动画延迟
    this.delay = 250
  }

  componentDidUpdate() {
    const mainNode = this.mainRef.current;
    if (this.state.isUnfold) {
      this.timer = setTimeout(() => {
        Array.from(mainNode.childNodes).map(item => item.style.display = '')
      }, this.delay)
    } else {
      clearTimeout(this.timer);
      this.timer = null;
      Array.from(mainNode.childNodes).map(item => item.style.display = 'none');
    }

    this.setState((prev, props) => {
      if (props.isShow === false) {
        Array.from(mainNode.childNodes).map(item => item.style.display = 'none');
        return {
          isUnfold: false
        }
      }
    })
  }

  render() {
    return (
      <div className='simple-bar'>
        <a
          onClick={(e) => {
            // 阻止默认事件
            e.preventDefault();
            // 点击时隐藏 Tooltip，之后延迟恢复 Tooltip
            this.setState({
              tooltipVisible: false
            })
            const timer = setTimeout(() => {
              this.setState({
                tooltipVisible: true
              })
            }, this.delay)
            this.setState({
              timer,
            })
            // 更新展开状态
            this.setState({
              isUnfold: !this.state.isUnfold,
            });
            this.props.callback?.();
          }}
          onMouseEnter={() => {
            this.setState({
              tooltipVisible: true,
            })
          }}
          onMouseLeave={() => {
            clearTimeout(this.state.timer)
            this.setState({
              tooltipVisible: false,
            })
          }}
        >
          <div className='head'>
            <this.IconFont
              type={this.props.iconType}
            />
          </div>
        </a>
        <div
          className='content'
          style={{
            width: this.state.isUnfold ? (this.props.width + 'px') : '0px'
          }}
        >
          <div
            className='main'
            style={{ width: this.props.width + 'px' }}
            ref={this.mainRef}
          >
            {this.props.children}
          </div>
        </div>
        <Tooltip placement='right' title={this.props.title} visible={this.state.tooltipVisible}>
          <div
            className='end'
            style={{
              left: this.state.isUnfold ? ((this.props.width) + 'px') : '0px'
            }}
          ></div>
        </Tooltip>
      </div>
    )
  }
}
