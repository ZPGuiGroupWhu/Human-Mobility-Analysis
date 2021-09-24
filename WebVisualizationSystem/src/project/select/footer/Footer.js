import React, { Component } from 'react';
import DeckGL from '@deck.gl/react';
import "./Footer.scss";
import {Card, Col, Row ,Pagination,Popover,Tooltip} from 'antd';
import {ArcLayer,GeoJsonLayer} from '@deck.gl/layers';
import ODs from './ODs.json'
import ShenZhen from './ShenZhen.json'
import $ from 'jquery';
class Footer extends Component {
  constructor(props) {
    super(props);
    this.pageSize=6;
    this.state = {
      currentPage:1,
      minValue: 0,
      maxValue: this.pageSize,
    }
  }
  componentWillUpdate(nextProps, nextState) {//当界面更新时删除对应canvas的上下文，防止Oldest context will be lost
    if(this.state.minValue!==nextState.minValue){//仅当用户换页时进行该操作
    for(let i=0;i<$("canvas[id='deckgl-overlay']").length;i++)
    {
      let gl = $("canvas[id='deckgl-overlay']")[i].getContext('webgl2');
      gl.getExtension('WEBGL_lose_context').loseContext();
    }
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
        minValue: (page-1) * this.pageSize,
        maxValue: page*this.pageSize
      });
    }
    this.setState({
      currentPage: page,
    });
  };

  render() {
    let data = ODs;
    return (
      <div className="outer-container">
      <div className="select-footer-ctn">
      <Row gutter={[8,8]} style={{width:"100%"}}>
      {data &&
          data.length > 0 &&
          data.slice(this.state.minValue, this.state.maxValue).map(val => (
            <Col span={24} key={val.id}>
            <Popover 
            title={val.id} trigger="hover" placement="left" 
            content={
              <div>
                <p>指标1</p>
                <p>指标2</p>
                <p>etc.</p>
              </div>
            } >
            <Card
              title={val.id}
              hoverable={true}
              size="small"
              bodyStyle={{padding:1}}
            >
            <div style={{height:"70px",position:"relative"}} >
            {/* 出现了新的问题，当使用Deck.gl时，会导致WARNING: Too many active WebGL contexts. Oldest context will be lost，从而使底图消失 */}
            {/* 问题已解决，在更新前删除对应canvas的上下文即可，保留备注以备不时之需*/}
              <DeckGL
                initialViewState={{
                longitude: 114.18,
                latitude: 22.7,
                zoom: 6.5,
                pitch: 45,
                bearing: 0}}
                controller={false}
                layers={[
                  new ArcLayer({
                    id: 'arc-layer',
                    data:val.ODs,
                    pickable: false,
                    getWidth: 2,
                    getSourcePosition: d => d.O,
                    getTargetPosition: d => d.D,
                    getSourceColor:  [255,250, 97],
                    getTargetColor:  [30, 20, 255],
                  }),
                  new GeoJsonLayer({
                    id: 'ShenZHen',
                    data:ShenZhen,
                    lineWidthMinPixels: 1,
                    getFillColor:[255,255,255]
                  })
                ]}>
              </DeckGL>
              </div>
            </Card>
            </Popover>
            </Col>
          ))}
        <Col span={24} key={"Pagination"}>
        <Pagination style={{position:"relative",left: "0%",top:"2%",transform:"translate(0%, 0)",width:"100%",textAlign:"center",backgroundColor:"white"}}
          simple size="small" current={this.state.currentPage} onChange={this.onChange} total={data.length} showSizeChanger={false}
          defaultPageSize={this.pageSize} />
        </Col>
      </Row>
      </div>
      </div>
    );
  }
}

export default Footer;