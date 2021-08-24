import React, { Component,useState} from 'react';
import DeckGL from '@deck.gl/react';
import {StaticMap} from 'react-map-gl';
import {HeatmapLayer,GPUGridLayer} from '@deck.gl/aggregation-layers';
import {ArcLayer, IconLayer} from '@deck.gl/layers';
import {TripsLayer} from '@deck.gl/geo-layers';
import {Switch,Slider} from 'antd';
import './DeckGLMap.css'
import getFlatternDistance from './distanceCalculater.js'
import { eventEmitter } from '@/common/func/EventEmitter';
import _ from 'lodash';
import Store from '@/store';

const MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiMjAxNzMwMjU5MDE1NyIsImEiOiJja3FqM3RjYmIxcjdyMnhsbmR0bHo2ZGVpIn0.wNBmzyxhzCMx9PhIH3rwCA';//MAPBOX密钥

class DeckGLMap extends Component {
  constructor(props){
    super(props);
    this.trajNodes=[];//轨迹点集合
    this.speedNodes=[];//速度点集合
    this.OdNodes = [];
    this.arcLayerShow=false;//是否显示OD弧段图层
    this.heatMapLayerShow=false;//是否显示热力图图层
    this.gridLayerShow=false;//是否显示格网图层
    this.gridLayer3D=false;//格网图层是否为3D
    this.speedLayerShow=false;//是否显示速度图层
    this.speedLayer3D=false;//速度图层是否为3D
    this.gridWidth=100;//格网图层的宽度
    this.tripsLayerShow=false;
    this.iconLayerShow=false;
    this.tripsLayerOneShow=false;
    this.arcLayerOneShow=false;
    this.state={
      arcLayer:null,//OD弧段图层
      heatMapLayer:null,//热力图图层
      gridLayer:null,//格网图层
      trajCounts:[],//每天的轨迹数目
      hoveredMessage: null,//悬浮框内的信息
      pointerX: null,//悬浮框的位置
      pointerY: null,
      selectDate: {
        start: '2018-01-01',
        end: '2018-12-31',
      },
      tripsLayer: null,//轨迹图层
      iconLayer: null,//icon图标图层
      arcLayerOne: null,//选中OD图层
      tripsLayerOne: null,//选中轨迹图层
      Opacity:0.8,
    }
  }

  componentDidMount() {
    this.getLayers();
    this.addDateSelectListener();
  }
  componentDidUpdate(prevProps, prevState) {
    if (prevProps.userData !== this.props.userData) {
      this.getLayers();
    }
    if (!_.isEqual(prevState.selectDate, this.state.selectDate)) {
      this.showSelectTraj(this.state.selectDate.start, this.state.selectDate.end);
    }
  }

  //获取所有的OD点
  getAllOdNodes = () => {
    let OdNodes = [];
    for (let i = 0; i < this.props.userData.length; i++) {
      OdNodes.push({COORDINATES: this.props.userData[i].O, id: this.props.userData[i].id});
      OdNodes.push({COORDINATES: this.props.userData[i].D, id: this.props.userData[i].id});
    }
    this.OdNodes = OdNodes;
  };

  getTrajNodes = () => {
    let Nodes=[];//统计所有节点的坐标
    let Count ={};//统计每天的轨迹数目
    let Speeds=[];//统计速度
    for(let i=0;i<this.props.userData.length;i++){
      if(Count[this.props.userData[i].date] === undefined){
        Count[this.props.userData[i].date]={'count':1}//若当天没有数据，则表明是第一条轨迹被录入
      }
      else{
        Count[this.props.userData[i].date]={'count':Count[this.props.userData[i].date].count+1}//否则是其他轨迹被录入，数目加一
      }
      for(let j=0;j<this.props.userData[i].data.length;j++){
        Nodes.push({COORDINATES:this.props.userData[i].data[j],WEIGHT:1});//将所有轨迹点放入同一个数组内，权重均设置为1
      }
      for(let j=2;j<this.props.userData[i].data.length;j++){
        let xy1=this.props.userData[i].data[j-1];
        let xy2=this.props.userData[i].data[j];
        let speed= getFlatternDistance(xy1[1],xy1[0],xy2[1],xy2[0]);
        if(speed>100)continue;
        Speeds.push({COORDINATES:this.props.userData[i].data[j],WEIGHT:speed});//假设两点间时间一致，速度用欧氏距离代替
      }
    }
    this.trajNodes=Nodes;
    this.trajCounts=Count;
    this.speedNodes=Speeds;
  };
  toParent = () => {//将每天的轨迹数目统计结果反馈给父组件
    this.props.getTrajCounts(this.trajCounts)
  };
  // 根据日期筛选可视化的轨迹
  getSelectData = (start, end) => {
    let selectOdNodes = [];
    let selectTrajs = [];
    let startTimeStamp = Date.parse(start);
    let endTimeStamp = Date.parse(end);
    for (let i = 0; i < this.props.userData.length; i++) {
      if (startTimeStamp <= Date.parse(this.props.userData[i].date) && Date.parse(this.props.userData[i].date) <= endTimeStamp){
        selectOdNodes.push({COORDINATES: this.props.userData[i].O, id: this.props.userData[i].id});
        selectOdNodes.push({COORDINATES: this.props.userData[i].D, id: this.props.userData[i].id});
        selectTrajs.push(this.props.userData[i]);
      }
    }
    // console.log(selectTrajs);
    return [selectOdNodes, selectTrajs]
  };
  //构建OD弧段图层
  getArcLayer = () =>{
    this.setState({
      arcLayer:new ArcLayer({
        id: 'arc-layer',
        data:this.props.userData,
        pickable: true,
        getWidth: 3,
        getSourcePosition: d => d.O,
        getTargetPosition: d => d.D,
        getSourceColor:  [255,250, 97],
        getTargetColor:  [30, 20, 255],})
    });
  };
  //构建热力图图层
  getHeatMapLayer = () =>{
    this.setState({
      heatMapLayer:new HeatmapLayer({
        id: 'heatmapLayer',
        data:this.trajNodes,
        getPosition: d => d.COORDINATES,
        getWeight: d => d.WEIGHT,
        aggregation: 'SUM'
      })
    })
  };
  //TODO 格网图层的色带有待修改，目前展示效果并不好
  getGridLayer = () =>{//构建格网图层
    this.setState({
      gridLayer:new GPUGridLayer({
        id: 'gpu-grid-layer',
        data:this.trajNodes,
        pickable: true,
        extruded: this.gridLayer3D,//是否显示为3D效果
        cellSize: this.gridWidth,//格网宽度，默认为100m
        elevationScale: 4,
        getPosition: d => d.COORDINATES,
        onHover:({object, x, y})=>{//构建悬浮框信息
          var str="";
          if(object==null){
            str=""
          }
          else{
            str='Count : '+object.count
          }
          this.setState({
            hoveredMessage: str,
            pointerX: x,
            pointerY: y,
          });
        }
      })
    })
  };
  //构建速度图层
  getSpeedLayer = () =>{
    this.setState({
      speedLayer:new GPUGridLayer({
        id: 'gpu-grid-layer-speed',
        data:this.speedNodes,
        pickable: true,
        extruded: this.speedLayer3D,//是否显示为3D效果
        cellSize: this.gridWidth,//格网宽度，默认为100m
        elevationScale: 4,
        elevationAggregation:'MEAN',//选用速度均值作为权重
        colorAggregation:'MEAN',
        colorRange:[[219, 251, 255],[0, 161, 179],[82, 157, 255],[0, 80, 184],[173, 66, 255],[95, 0, 168]],
        getPosition: d => d.COORDINATES,
        getElevationWeight:d=>d.WEIGHT,
        getColorWeight:d=>d.WEIGHT,
        onHover:({object, x, y})=>{//构建悬浮框信息
          var str="";
          if(object==null){
            str=""
          }
          else{
            str='Speed : '+object.elevationValue
          }
          this.setState({
            hoveredMessage: str,
            pointerX: x,
            pointerY: y,
          });
        }
      })
    })
  };

  //轨迹点击事件
  clickTraj = (info) =>{
    this.setState({
      Opacity:0.02
    }, ()=>{
      let id = info ? info.object.id : "not found";
      // 绘制OD弧线
      const tempOD = [];
      const tempTraj = [];
      for (let i = 0; i < this.props.userData.length; i++){
        if (this.props.userData[i].id === id){
          tempOD.push({O:this.props.userData[i].O, D:this.props.userData[i].D});
          for(let j = 0; j < this.props.userData[i].data.length; j++){
            tempTraj.push(this.props.userData[i].data[j])
          }
          break
        }
      }
      const tempPath = [{path: tempTraj}];
      this.setState({
        arcLayerOne:new ArcLayer({
          id: 'arc-layer-one',
          data: tempOD,
          pickable: true,
          getWidth: 1,
          getSourcePosition: d => d.O,
          getTargetPosition: d => d.D,
          getSourceColor: [175, 255, 255],
          getTargetColor: [0, 128, 128],
        })
      });
      //轨迹高亮
      // console.log(tempPath);
      this.setState({
        tripsLayerOne: new TripsLayer({
          id: 'trips-layer-one',
          data: tempPath,
          getPath: d => d.path,
          // deduct start timestamp from each data point to avoid overflow
          // getTimestamps: d => d.waypoints.map(p => p.timestamp - 1554772579000),
          getColor: [256, 0, 0],
          opacity: 1,
          widthMinPixels: 3,
          rounded: true,
          trailLength: 200,
          currentTime: 100,
        })
      });
      // 存储轨迹
      this.context.dispatch({ type: 'setSelectedTraj', payload: info.object });
      this.showSelectTraj(this.state.selectDate.start, this.state.selectDate.end);
    });
  };

  // 绘制轨迹图层
  getTripsLayer = (selectData) =>{
    this.setState({
      tripsLayer: new TripsLayer({
        id: 'trips-layer',
        data: selectData,
        getPath: d => d.data,
        // deduct start timestamp from each data point to avoid overflow
        // getTimestamps: d => d.waypoints.map(p => p.timestamp - 1554772579000),
        getColor: [244,164,96],
        // opacity: 512/ this.state.dataLength,
        opacity: this.state.Opacity,
        widthMinPixels: 3,
        rounded: true,
        fadeTrail: false,
        trailLength: 200,
        currentTime: 100,
        pickable: true,
        autoHighlight: true,
        highlightColor: [256, 0, 0, 256],
        onClick: info => this.clickTraj(info),
      })
    })
  };

  // OD图层
  getIconLayer = (selectData) => {
    const ICON_MAPPING = {
      marker: {x: 0, y: 0, width: 128, height:128, mask: true}
    };
    this.setState(
        {
          iconLayer: new IconLayer({
            id: 'icon-layer',
            data:selectData,
            pickable: true,
            // iconAtlas and iconMapping are required
            // getIcon: return a string
            iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
            iconMapping: ICON_MAPPING,
            getIcon: d => 'marker',
            // getIcon: d => ({
            //   url: "E:/workspace/deckglmap/src/location-icon-atlas.png",
            //   width: 128,
            //   height: 128,
            // }),
            sizeScale: 10,
            onClick: info => this.clickTraj(info),
            getPosition: d => d.COORDINATES,
            getSize: d => 2,
            getColor: d => [255, 140, 0],
          })
        }
    )
  };

  getArcLayerOne = () =>{
    this.setState({
      arcLayerOne: null,
    })
  };

  getTripsLayerOne = () =>{
    this.setState({
      tripsLayerOne: null,
    })
  };

  // 可视化筛选的轨迹
  showSelectTraj = (start, end) =>{
    this.setState({
      Opacity:0.8
    });
    const [selectOdNodes, selectTrajs] = this.getSelectData(start, end);
    this.getIconLayer(selectOdNodes);
    this.getTripsLayer(selectTrajs);
  };

  changeGridLayerShow=()=>{//与开关联动，切换格网图层的显示与否
    this.gridLayerShow=!this.gridLayerShow;
    this.getGridLayer();
  };
  changeGridLayer3D=()=>{//与开关联动，切换格网图层的3D效果
    this.gridLayer3D=!this.gridLayer3D;
    this.getGridLayer();
  };
  changeSpeedLayerShow=()=>{//与开关联动，切换速度图层的显示与否
    this.speedLayerShow=!this.speedLayerShow;
    this.getSpeedLayer();
  };
  changeSpeedLayer3D=()=>{//与开关联动，切换速度图层的3D效果
    this.speedLayer3D=!this.speedLayer3D;
    this.getSpeedLayer();
  };
  changeGridWidth=(value)=>{//与滑动条联动，切换格网的网格宽度
    this.gridWidth=value;
    this.getGridLayer();
    this.getSpeedLayer();
  };
  changeHeatMapLayerShow=()=>{//与开关联动，切换热力图图层的显示与否
    this.heatMapLayerShow=!this.heatMapLayerShow;
    this.getHeatMapLayer();
  };
  changeArcLayerShow=()=>{//与开关联动，切换OD弧段图层的显示与否
    this.arcLayerShow=!this.arcLayerShow;
    this.getArcLayer();
  };
  changeTripsLayerShow=()=>{//与开关联动，切换轨迹图层和icon图层的显示与否
    // 初始化透明度
    this.setState({
      Opacity:0.8
    });
    // this.iconLayerShow=!this.iconLayerShow;
    this.tripsLayerShow=!this.tripsLayerShow;
    this.tripsLayerOneShow=!this.tripsLayerOneShow;
    this.arcLayerOneShow=!this.arcLayerOneShow;
    this.getTripsLayerOne();
    this.getArcLayerOne();
    // this.getIconLayer(this.OdNodes);
    this.getTripsLayer(this.props.userData);
  };

  getLayers = () => {//获取所有图层
    this.getTrajNodes();//获取所有轨迹点的集合
    this.getArcLayer();//构建OD弧段图层
    this.getHeatMapLayer();//构建热力图图层
    this.getGridLayer();//构建格网图层
    this.getSpeedLayer();//构建速度图层
    this.toParent();//将每天的轨迹数目统计结果反馈给父组件
    this.getAllOdNodes();//获取所有的OD集
    this.getTripsLayer(this.props.userData);//构建轨迹图层
    this.getIconLayer(this.OdNodes);//构建icon图标图层
    this.getTripsLayerOne();//初始化单挑高亮轨迹图层
    this.getArcLayerOne();//初始化单挑OD图层
  };
  // 根据日期选择轨迹，监听函数
  addDateSelectListener() {
    eventEmitter.on(this.props.eventName, ({ start, end }) => {
      this.setState({
        selectDate: {
          start,
          end,
        },
        arcLayerOne: null,
        tripsLayerOne: null
      })
    })
  };
  _renderTooltip() {//TooTip的渲染
    const {hoveredMessage, pointerX, pointerY} = this.state || {};
    return hoveredMessage && (
        <div style={{position: 'absolute', zIndex: 999, pointerEvents: 'none', left: pointerX, top: pointerY, color: '#fff', backgroundColor: 'rgba(100,100,100,0.5)',"whiteSpace": "pre"}}>
          { hoveredMessage }
        </div>
    );
  }
  render(){
    return(
        <div>
          <DeckGL
              initialViewState={{
                longitude: 114.17,
                latitude: 22.65,
                zoom: 10,
                pitch: 45,
                bearing: 0}}
              controller={true}
              layers={[this.gridLayerShow?this.state.gridLayer:null,
                this.heatMapLayerShow?this.state.heatMapLayer:null,
                this.arcLayerShow?this.state.arcLayer:null,
                this.speedLayerShow?this.state.speedLayer:null,
                this.tripsLayerShow?this.state.tripsLayer:null,
                this.iconLayerShow?this.state.iconLayer:null,
                this.arcLayerOneShow?this.state.arcLayerOne:null,
                this.tripsLayerOneShow?this.state.tripsLayerOne:null,
              ]}>
            <StaticMap mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN} mapStyle={'mapbox://styles/2017302590157/cksbi52rm50pk17npkgfxiwni'}/>
            { this._renderTooltip() }
          </DeckGL>
          <div className={`moudle`}>
            GridLayer   <Switch onChange={this.changeGridLayerShow}/><br />
            GridLayer3D <Switch onChange={this.changeGridLayer3D}/><br />
            SpeedLayer   <Switch onChange={this.changeSpeedLayerShow}/><br />
            SpeedLayer3D <Switch onChange={this.changeSpeedLayer3D}/><br />
            HeatMapLayer<Switch  onChange={this.changeHeatMapLayerShow} /><br />
            ArcLayer    <Switch  onChange={this.changeArcLayerShow} /><br />
            TripsLayer   <Switch  onChange={this.changeTripsLayerShow} /><br />
          </div><br/>
          <div className={`moudle`} style = {{'textAlign':'center'}}>
            GridWidth   <Slider style = {{width:'140px'}} max={500} min={50} step={50} defaultValue={100} onChange={(value) => this.changeGridWidth(value)}/>
          </div>
        </div>
    )
  }
}

DeckGLMap.contextType = Store;

export default DeckGLMap