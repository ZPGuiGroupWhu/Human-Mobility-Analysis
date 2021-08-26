import React, { Component } from 'react';
import './Box.scss';
import { CompressOutlined, ExpandOutlined, CloseOutlined } from '@ant-design/icons';
import { Space } from 'antd';
import { CSSTransition } from 'react-transition-group';

class Box extends Component {
  // 静态配置项
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  constructor(props) {
    super(props);
    // props
    this.title = props.title;
    // state
    this.state = {
      boxVisible: true,
      chartVisible: true,
    };
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      chartVisible: !prev.chartVisible
    }))
  }

  // box展示/隐藏
  setBoxVisible = () => {
    this.setState(prev => ({
      boxVisible: !prev.boxVisible,
      chartVisible: prev.boxVisible ? false : true,
    }))
  }


  componentDidUpdate(prevProps, prevState) {
  }

  render() {
    return (
      <>
        <div className="chart-box-ctn" style={this.state.boxVisible ? null : { width: '80px' }}>
          <div className="title-bar">
            {
              this.state.boxVisible ?
              (
                <>
                  <span className="text">{this.title}</span>
                  <div className="func-btns">
                    <Space>
                      {
                        this.state.chartVisible ?
                          <CompressOutlined style={this.iconStyle} onClick={this.setChartVisible} /> :
                          <ExpandOutlined style={this.iconStyle} onClick={this.setChartVisible} />
                      }
                      <CloseOutlined style={this.iconStyle} onClick={this.setBoxVisible} />
                    </Space>
                  </div>
                </>
              ) :
              (
                <span className="text" onClick={this.setBoxVisible}>{this.title}</span>
              )
            }
          </div>
          <CSSTransition
            in={this.state.chartVisible}
            timeout={300}
            classNames='chart'
            onEnter={(node) => { node.style.setProperty('display', '') }}
            onExiting={(node) => { node.style.setProperty('display', 'none') }}
          >
            <div
              className="chart-content"
            >
              {this.props.children}
            </div>
          </CSSTransition>
        </div>
      </>

    );
  }
}

export default Box;