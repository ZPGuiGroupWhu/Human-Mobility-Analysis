import React, { Component } from 'react';
import './layout.scss';


export const MouseXY = React.createContext([0, 0])

class Layout extends Component {
  constructor(props) {
    super(props);
    this.state = {
      mouseXY: [0, 0]
    }
  }

  render() {
    const {
      src,
      title,
      imgWidth, // 可缺省，不建议更改图片宽度，缺省可保证图片宽高默认比
      imgHeight, // 可缺省，调整图片高度百分比，可实现图片大小一定范围内更改
    } = this.props;
    const children = React.Children.toArray(this.props.children)

    return (
      <div className='geo-container'>
        <div className='header'>
          <div className='img-title'>
            <img
              src={src}
              alt=''
              style={{ width: imgWidth, height: imgHeight }}
            />
            <h1>{title}</h1>
          </div>
          {/* <div className='func-bar'>
            {children[0]}
          </div> */}
        </div>
        <div className='main' onMouseMove={e => { this.setState({ mouseXY: [e.pageX, e.pageY] }) }}>
          <MouseXY.Provider value={this.state.mouseXY}>
            {children[0]}
          </MouseXY.Provider>
        </div>
      </div>
    )
  }
}

export default Layout;