import React, { Component } from 'react';
import DeckGL from '@deck.gl/react';
import "./Footer.scss";
import {Card, Col, Row ,Pagination,Popover} from 'antd';
import {ArcLayer,GeoJsonLayer} from '@deck.gl/layers';
import ODs from './ODs.json'
import ShenZhen from './ShenZhen.json'
import $ from 'jquery';
import Store from '@/store'
class Footer extends Component {
  static contextType = Store;
  constructor(props) {
    super(props);
    this.changeTimes=0;
    this.pageSize=8;
    this.data=ODs;
    this.state = {
      currentPage:1,
      minValue: 0,
      maxValue: this.pageSize,
    }
  }
  componentWillUpdate(nextProps, nextState) {//当界面更新时删除对应canvas的上下文，防止Oldest context will be lost
    this.data=ODs
    if(this.context.state.selectedUsers.length>0){
      this.data=[]
      for(let i=0;i<this.context.state.selectedUsers.length;i=i+1){
        this.data.push(ODs.find(item=>item.id==this.context.state.selectedUsers[i]))
      }
    }
    if(this.state.maxValue!==nextState.maxValue ||this.props.selectedByCharts !== nextProps.selectedByCharts || this.props.selectedByCalendar !== nextProps.selectedByCalendar){
      // console.log("clear canvas")
      this.changeTimes=this.changeTimes+1;
    for(let i=0;i<$("canvas[id='deckgl-overlay']").length;i++)
    {
      let gl = $("canvas[id='deckgl-overlay']")[i].getContext('webgl2');
      gl.getExtension('WEBGL_lose_context').loseContext();
    }
    }
  }
  componentDidUpdate(prevProps,prevState){
    if(this.props.selectedByCharts !== prevProps.selectedByCharts || this.props.selectedByCalendar !== prevProps.selectedByCalendar){
      this.setState({currentPage:1,minValue: 0,maxValue: this.pageSize})
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

  getPopInfo=(id)=>{
    let info=[]
    this.context.state.allData&&Object.keys(Object.values(this.context.state.allData).find(item=>item['人员编号']==id)).forEach(key=>(
      info.push(<li style={{float:"left",width:"50%"}}>{key}:{Object.values(this.context.state.allData).find(item=>item['人员编号']==id)[key]}</li>)
      ))
    info.push(<div className="clear"></div>)
    return info
  }
  render() {
    return (
      <div className="outer-container">
      <div className="select-footer-ctn">
      <Row gutter={[8,8]} style={{width:"100%"}}>
      <Col span={24} key={"Pagination"}>
        <Pagination style={{fontSize:12, position:"relative",left: "0%",top:"2%",transform:"translate(0%, 0)",width:"100%",textAlign:"center",backgroundColor:"white"}}
          simple size='small' current={this.state.currentPage} onChange={this.onChange} total={this.data.length} showSizeChanger={false}
          defaultPageSize={this.pageSize} />
      </Col>
      {this.data &&
          this.data.length > 0 &&
          this.data.slice(this.state.minValue, this.state.maxValue).map(val => (
            <Col span={24} key={this.changeTimes+'-'+val.id}>
            <Popover 
            title={val.id} trigger="click" placement="left" 
            content={
              <div style={{width:"500px"}}>
                {this.getPopInfo(val.id)}
              </div>
            } >
            <Card
              title={val.id}
              hoverable={false}
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
                getCursor={({isDragging})=> 'default'}
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
      </Row>
      </div>
      </div>
    );
  }
}

export default Footer;