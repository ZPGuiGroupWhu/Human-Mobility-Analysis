import React, { Component } from 'react';
import './layout.scss';


export const MouseXY = React.createContext([0, 0])

const logoRef = React.createRef();
class Layout extends Component {
  constructor(props) {
    super(props);
    this.state = {
      mouseXY: [0, 0]
    }
  }

  // logo animation
  setLogoAnimation = function (target) {
    let round = 0;
    const renderAnimation = () => {
      if (round > 360) round = 0;
      target.style.setProperty('transform', `rotate(${round++}deg)`);
      requestAnimationFrame(renderAnimation);
    }
    requestAnimationFrame(renderAnimation);
  }

  componentDidMount() {
    this.setLogoAnimation(logoRef.current)
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
              ref={logoRef}
              src={src}
              alt=''
              style={{ width: imgWidth, height: imgHeight }}
              draggable={false}
            />
            <h1>{title}</h1>
          </div>
          <div className='func-bar'>
            {children[0]}
          </div>
        </div>
        <div className='main' onMouseMove={e => { this.setState({ mouseXY: [e.pageX, e.pageY] }) }}>
          <MouseXY.Provider value={this.state.mouseXY}>
            {children[1]}
          </MouseXY.Provider>
        </div>
        {/* 分割线 */}
        <div className="split-line"></div>
      </div>
    )
  }
}

export default Layout;