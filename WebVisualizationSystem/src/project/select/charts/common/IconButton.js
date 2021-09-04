import React, { Component } from 'react';
import Hover from './Hover';

class IconButton extends Component {
  static defaultProps = {
    withClickArgs: [],
    imgHeight: '25px',
  }

  /**
   * props
   * @param {function} onClick - 点击事件函数
   * @param {array?} withClickArgs - 点击事件函数参数 
   * @param {string} actImage - 按钮激活的图片地址
   * @param {string} noActImage - 按钮未激活的图片地址
   * @param {string?} imgHeight - 图片高度
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

  onClick = (e, ...params) => {
    e.preventDefault();
    this.props.onClick.call(this, ...params);
  }

  render() {
    return (
      <Hover>
        {
          ({ isHovering, isClicked }) => (
            <a
              style={this.aStyle}
              onClick={(e) => this.onClick(e, ...this.props.withClickArgs)}
            >
              {
                (isHovering || isClicked) ?
                  <img src={this.props.actImage} alt="" style={{ height: this.props.imgHeight }} /> :
                  <img src={this.props.noActImage} alt="" style={{ height: this.props.imgHeight }} />
              }
            </a>
          )
        }
      </Hover>
    );
  }
}

export default IconButton;