import React, { Component } from 'react';
import DeckGL from '@deck.gl/react';
import { NavLink } from 'react-router-dom';
import "./Footer.scss";
import '@/project/border-style.scss'
import { Card, Col, Row, Pagination, Popover } from 'antd';
import { ArcLayer, GeoJsonLayer } from '@deck.gl/layers';
import ODs from './ODs.json'
import ShenZhen from './ShenZhen.json'
import $ from 'jquery';
import _ from 'lodash';
// react-redux
import { connect } from 'react-redux';
// 组建
import PopupContent from './components/popupContent/PopupContent';


class Footer extends Component {
  constructor(props) {
    super(props);
    this.changeTimes = 0;
    this.pageSize = 8;
    this.data = ODs;
    this.state = {
      currentPage: 1,
      minValue: 0,
      maxValue: this.pageSize,
    }
  }
  componentWillUpdate(nextProps, nextState) {//当界面更新时删除对应canvas的上下文，防止Oldest context will be lost
    this.data = ODs
    if (this.props.selectedUsers.length > 0) {
      this.data = []
      for (let i = 0; i < this.props.selectedUsers.length; i = i + 1) {
        this.data.push(ODs.find(item => item.id == this.props.selectedUsers[i]))
      }
    }
    if (
      this.props.selectedUsers !== nextProps.selectedUsers ||
      this.state.maxValue !== nextState.maxValue) {
      // console.log("clear canvas")
      this.changeTimes = this.changeTimes + 1;
      for (let i = 0; i < $("canvas[id='deckgl-overlay']").length; i++) {
        let gl = $("canvas[id='deckgl-overlay']")[i].getContext('webgl2');
        gl.getExtension('WEBGL_lose_context').loseContext();
      }
    }
  }
  componentDidUpdate(prevProps, prevState) {
    if (this.props.selectedUsers !== prevProps.selectedUsers) {
      this.setState({ currentPage: 1, minValue: 0, maxValue: this.pageSize })
    }
  }
  onChange = (page) => {
    if (page <= 1) {
      this.setState({
        minValue: 0,
        maxValue: this.pageSize
      });
    } else {
      this.setState({
        minValue: (page - 1) * this.pageSize,
        maxValue: page * this.pageSize
      });
    }
    this.setState({
      currentPage: page,
    });
  };

  getPopInfo = (id) => {
    let info = []
    this.props.data && Object.keys(Object.values(this.props.data).find(item => item['人员编号'] == id)).forEach(key => (
      info.push(<li style={{ float: "left", width: "50%" }}>{key}:{Object.values(this.props.data).find(item => item['人员编号'] == id)[key]}</li>)
    ))
    info.push(<div className="clear"></div>)
    return info
  }
  cardClick = (id, e) => {
    for (let i = 0; i < $("div[id='deckgl-card']").length; i++) {
      $("div[id='deckgl-card']")[i].style.backgroundColor = "";
    }
    if (e.target.className == "ant-card-head-title") {
      e.target.parentElement.parentElement.parentElement.style.backgroundColor = "#c7f0ff"
    }
    else if (e.target.className == "ant-card-body") {
      e.target.parentElement.style.backgroundColor = "#c7f0ff"
    }
    else if (e.target.id == "deckgl-overlay") {
      e.target.parentElement.parentElement.parentElement.parentElement.style.backgroundColor = "#c7f0ff"
    }
    this.props.setRoutes(prev => {
      const newRoutes = _.cloneDeep(prev);
      newRoutes[1].status = true;
      return newRoutes;
    })
  }
  render() {
    return (
      <div className="outer-container tech-border">
        <Row gutter={[8, 8]} style={{ width: "100%", marginLeft: "0" }}>
          <Col span={24} key={"Pagination"} style={{ marginBottom: "10px" }}>
            <Pagination style={{ fontSize: 12, position: "relative", left: "0%", top: "2%", transform: "translate(0%, 0)", width: "100%", textAlign: "center", backgroundColor: "white" }}
              simple size='small' current={this.state.currentPage} onChange={this.onChange} total={this.data.length} showSizeChanger={false}
              defaultPageSize={this.pageSize} />
          </Col>
        </Row>
        <div className="select-footer-ctn">
          <Row gutter={[8, 8]} style={{ width: "100%", marginLeft: "0" }}>
            {this.data &&
              this.data.length > 0 &&
              this.data.slice(this.state.minValue, this.state.maxValue).map(val => (
                <Col span={24} key={this.changeTimes + '-' + val.id}>
                  <Popover
                    overlayClassName='popover'
                    title={
                      <div>
                        <span className="user-id">{'用户ID：' + val.id}</span>
                        <NavLink
                          style={{ color: '#8e44ad', float: "right", fontWeight: 'bold' }}
                          activeStyle={{ color: '#15FBF1', fontWeight: 'bold', }}
                          to={location => {
                            return {
                              ...location,
                              pathname: '/select/analysis',
                            }
                          }
                          }
                          exact
                        >
                          轨迹筛选
                        </NavLink>
                      </div>
                    } trigger="click" placement="left"
                    content={
                      <PopupContent id={val.id} />
                      // <div style={{ width: "500px" }}>
                      //   {this.getPopInfo(val.id)}
                      // </div>
                    } >
                    <Card
                      title={val.id}
                      hoverable={false}
                      size="small"
                      bodyStyle={{ padding: 1 }}
                      id="deckgl-card"
                      // className="card"
                      // tabindex={val.id}
                      onClick={(e) => this.cardClick(val.id, e)}
                    >
                      <div style={{ height: "70px", position: "relative" }} >
                        {/* 出现了新的问题，当使用Deck.gl时，会导致WARNING: Too many active WebGL contexts. Oldest context will be lost，从而使底图消失 */}
                        {/* 问题已解决，在更新前删除对应canvas的上下文即可，保留备注以备不时之需*/}
                        <DeckGL
                          initialViewState={{
                            longitude: 114.18,
                            latitude: 22.7,
                            zoom: 6.5,
                            pitch: 45,
                            bearing: 0
                          }}
                          controller={false}
                          getCursor={({ isDragging }) => 'default'}
                          layers={[
                            new ArcLayer({
                              id: 'arc-layer',
                              data: val.ODs,
                              pickable: false,
                              getWidth: 2,
                              getSourcePosition: d => d.O,
                              getTargetPosition: d => d.D,
                              getSourceColor: [255, 250, 97],
                              getTargetColor: [30, 20, 255],
                            }),
                            new GeoJsonLayer({
                              id: 'ShenZHen',
                              data: ShenZhen,
                              lineWidthMinPixels: 1,
                              getFillColor: [255, 255, 255]
                            })
                          ]}>
                        </DeckGL>
                      </div>
                    </Card>
                  </Popover>
                </Col>
              ))}
          </Row>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state) => (
  {
    selectedUsers: state.select.selectedUsers,
    data: state.select.data,
  }
)

export default connect(mapStateToProps, null)(Footer);