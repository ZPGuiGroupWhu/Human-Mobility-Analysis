import scssStyle from './InfoBar.scss';
import React, { PureComponent } from 'react';
import { createFromIconfontCN } from '@ant-design/icons';
import { Tooltip } from 'antd';

export default class SimpleBar extends PureComponent {
  timer = null;

  constructor(props) {
    super(props);
    this.state = {
      timer: null, // 计时器
      histState: false, // 上一时刻展开状态
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

  componentDidUpdate(prevProps, prevState) {
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

    // 跟随侧栏调整当前展开状态，侧栏收缩，则全部折叠；侧栏展开，则恢复历史展开状态
    this.setState((prev, props) => {
      if (props.isShow === false) {
        Array.from(mainNode.childNodes).map(item => item.style.display = 'none');
        return {
          histState: prevState.isUnfold,
          isUnfold: false
        }
      }
      if (!Object.is(prevProps.isShow, props.isShow)) {
        if (prev.histState) {
          this.timer = setTimeout(() => {
            Array.from(mainNode.childNodes).map(item => item.style.display = '')
          }, this.delay)
        }
        return {
          isUnfold: prevState.histState
        }
      }
    })
  }

  render() {
    return (
      <div className='info-bar'>
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
              histState: this.state.isUnfold,
              isUnfold: !this.state.isUnfold
            });
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
          <Tooltip
            placement='right'
            title={this.props.title}
            visible={!this.state.isUnfold && this.state.tooltipVisible}
          >
            <div className='head'>
              <this.IconFont
                type={this.props.iconType}
              />
            </div>
          </Tooltip>

        </a>

        <div
          className='background'
          style={{
            width: this.state.isUnfold ? (this.props.width + 'px') : scssStyle.height,
            height: this.state.isUnfold ? (this.props.height + 'px') : scssStyle.height,
          }}
        >
          <div className='content' ref={this.mainRef} style={{display: this.state.isUnfold ? '' : 'none'}}>
            {this.props.children}
          </div>
        </div>
      </div>
    )
  }
}
