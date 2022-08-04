import React, { Component } from 'react';
import { CSSTransition } from 'react-transition-group';
import './MapSelectWindow.scss';
import '@/project/border-style.scss';
import {
    CompressOutlined,
    ExpandOutlined,
} from '@ant-design/icons';
import { Space } from 'antd';
import Hover from '../../charts/common/Hover';
import MapSelectBar from './MapSelectBar';

class MapSelectWindow extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isVisible: true
        }
    }

    // 内容展开/关闭
    setChartVisible = () => {
        this.setState(prev => ({
            isVisible: !prev.isVisible
        }))
    }

    render() {
        return (
            <div
                className="map-select-ctn tech-border"
                style={{
                    right: this.props.right + 5,
                    bottom: this.props.bottom + 20,
                    height: this.state.isVisible ? 200 : 40
                }}>
                <div className='title-bar'>
                    <div className='map-box-title'>
                        {
                            this.state.isVisible ?
                                <span style={{ color: '#fff', fontFamily: 'sans-serif', fontSize: '15px', fontWeight: 'bold' }}>{'地图框选功能'}</span>
                                :
                                <span style={{ color: '#fff', fontFamily: 'sans-serif', fontSize: '15px', fontWeight: 'bold' }}>{'2019中国深圳市行政区地图'}</span>
                        }
                    </div>
                    <div className='map-box-switch'>
                        <Space>
                            {
                                this.state.isVisible ?
                                    <Hover>
                                        {
                                            ({ isHovering }) => (
                                                <CompressOutlined
                                                    style={{
                                                        ...this.iconStyle,
                                                        color: isHovering ? '#05f8d6' : '#fff'
                                                    }}
                                                    onClick={this.setChartVisible}
                                                />
                                            )
                                        }
                                    </Hover>
                                    :
                                    <Hover>
                                        {
                                            ({ isHovering }) => (
                                                <ExpandOutlined
                                                    style={{
                                                        ...this.iconStyle,
                                                        color: isHovering ? '#05f8d6' : '#fff'
                                                    }}
                                                    onClick={this.setChartVisible}
                                                />
                                            )
                                        }
                                    </Hover>
                            }
                        </Space>
                    </div>
                </div>
                <CSSTransition
                    in={this.state.isVisible}
                    timeout={300}
                    classNames='chart'
                    onEnter={(node) => { node.style.setProperty('display', '') }}
                    onExiting={(node) => { node.style.setProperty('display', 'none') }}
                >
                    <div className="chart-content">
                        <MapSelectBar />
                    </div>
                </CSSTransition>
            </div>
        );
    }
}

export default MapSelectWindow;