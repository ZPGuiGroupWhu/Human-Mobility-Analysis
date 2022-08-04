import React, { Component } from 'react';
import Hover from './Hover';
import { Tooltip } from 'antd';

class IconButton extends Component {
  static defaultProps = {
    withClickArgs: [],
    imgHeight: '25px',
    isReserveActive: true,
    isClicked: false,
  }

  /**
   * props
   * @param {function} onClick - 点击事件函数
   * @param {array?} withClickArgs - 点击事件函数参数 
   * @param {string} actImage - 按钮激活的图片地址
   * @param {string} noActImage - 按钮未激活的图片地址
   * @param {string?} imgHeight - 图片高度
   * @param {boolean} isReserveActive - 点击后是否保留激活效果
   * @param {string} text - 提示框文本内容
   * @param {boolean} isClicked - 是否默认点击
   */
  constructor(props) {
    super(props);
    this.state = {}
  }

  aStyle = {
    cursor: 'pointer',
    text_decoration: 'none',
    outline: 'none',
    color: '#000',
  }

  commonStyle = {
    transition: 'all .3s ease-out',
  }

  onClick = (e, ...params) => {
    e.preventDefault();
    this.props.onClick.call(this, ...params);
  }

  render() {
    return (
      <Hover isClicked={this.props.isClicked}>
        {
          ({ isHovering, isClicked }) => (
            <a
              style={this.aStyle}
              onClick={(e) => this.onClick(e, ...this.props.withClickArgs)}
            >
              {
                (isHovering || (this.props.isReserveActive && isClicked)) ?
                  <Tooltip placement="topLeft" title={<span>{this.props.text}</span>} color='cyan'>
                    <img
                      src={this.props.actImage}
                      alt=""
                      style={{
                        height: this.props.imgHeight,
                        transform: 'rotate(180deg)',
                        ...this.commonStyle
                      }}
                    />
                  </Tooltip>
                  :
                  <Tooltip placement="topLeft" title={<span>{this.props.text}</span>} color='cyan'>
                    <img src={this.props.noActImage} alt="" style={{ height: this.props.imgHeight, ...this.commonStyle }} />
                  </Tooltip>

              }
            </a>
          )
        }
      </Hover>
    );
  }
}

export default IconButton;