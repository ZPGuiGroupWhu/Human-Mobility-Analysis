import React, { Component,useState} from 'react';
import DeckGL from '@deck.gl/react';
import {StaticMap} from 'react-map-gl';
import {HeatmapLayer,GPUGridLayer} from '@deck.gl/aggregation-layers';
import {ArcLayer} from '@deck.gl/layers';
import {Switch,Slider} from 'antd';
import './DeckGLMap.css'
import getFlatternDistance from './distanceCalculater.js'

const MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiMjAxNzMwMjU5MDE1NyIsImEiOiJja3FqM3RjYmIxcjdyMnhsbmR0bHo2ZGVpIn0.wNBmzyxhzCMx9PhIH3rwCA';//MAPBOX密钥

class DeckGLMap extends Component {
    constructor(props){
        super(props);
        this.trajNodes=[];//轨迹点集合
        this.speedNodes=[];//速度点集合
        this.arcLayerShow=false;//是否显示OD弧段图层
        this.heatMapLayerShow=false;//是否显示热力图图层
        this.gridLayerShow=true;//是否显示格网图层
        this.gridLayer3D=true;//格网图层是否为3D
        this.speedLayerShow=false;//是否显示速度图层
        this.speedLayer3D=false;//速度图层是否为3D
        this.gridWidth=100;//格网图层的宽度
        this.state={
            arcLayer:null,//OD弧段图层
            heatMapLayer:null,//热力图图层
            gridLayer:null,//格网图层
            trajCounts:[],//每天的轨迹数目
            hoveredMessage: null,//悬浮框内的信息
            pointerX: null,//悬浮框的位置
            pointerY: null,
        }
    }
    componentDidMount(){
        this.getLayers();
    }
    componentDidUpdate(prevProps){
        if(prevProps.userData!==this.props.userData){
            this.getLayers();
        }
    }
    getTrajNodes = () => {
        let Nodes=[]//统计所有节点的坐标
        let Count ={}//统计每天的轨迹数目
        let Speeds=[]//统计速度
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
                let xy1=this.props.userData[i].data[j-1]
                let xy2=this.props.userData[i].data[j]
                let speed= getFlatternDistance(xy1[1],xy1[0],xy2[1],xy2[0])
                if(speed>100)continue;
                Speeds.push({COORDINATES:this.props.userData[i].data[j],WEIGHT:speed});//假设两点间时间一致，速度用欧氏距离代替
            }
        }
        this.trajNodes=Nodes;
        this.trajCounts=Count;
        this.speedNodes=Speeds;
        // console.log(Speeds)
    }
    toParent = () => {//将每天的轨迹数目统计结果反馈给父组件
        this.props.getTrajCounts(this.trajCounts)
    }
    getArcLayer = () =>{//构建OD弧段图层
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
    }
    getHeatMapLayer = () =>{ //构建热力图图层
        this.setState({
            heatMapLayer:new HeatmapLayer({
                id: 'heatmapLayer',
                data:this.trajNodes,
                getPosition: d => d.COORDINATES,
                getWeight: d => d.WEIGHT,
                aggregation: 'SUM'
            })
        })
    }
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
                    var str=""
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
    }
    getSpeedLayer = () =>{//构建速度图层
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
                    var str=""
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
    }
    changeGridLayerShow=()=>{//与开关联动，切换格网图层的显示与否
        this.gridLayerShow=!this.gridLayerShow;
        this.getGridLayer();
    }
    changeGridLayer3D=()=>{//与开关联动，切换格网图层的3D效果
        this.gridLayer3D=!this.gridLayer3D;
        this.getGridLayer();
    }
    changeSpeedLayerShow=()=>{//与开关联动，切换速度图层的显示与否
        this.speedLayerShow=!this.speedLayerShow;
        this.getSpeedLayer();
    }
    changeSpeedLayer3D=()=>{//与开关联动，切换速度图层的3D效果
        this.speedLayer3D=!this.speedLayer3D;
        this.getSpeedLayer();
    }
    changeGridWidth=(value)=>{//与滑动条联动，切换格网的网格宽度
        this.gridWidth=value;
        this.getGridLayer();
        this.getSpeedLayer();
    }
    changeHeatMapLayerShow=()=>{//与开关联动，切换热力图图层的显示与否
        this.heatMapLayerShow=!this.heatMapLayerShow;
        this.getHeatMapLayer();
    }
    changeArcLayerShow=()=>{//与开关联动，切换OD弧段图层的显示与否
        this.arcLayerShow=!this.arcLayerShow;
        this.getArcLayer();
    }
    getLayers = () => {//获取所有图层
        this.getTrajNodes();//获取所有轨迹点的集合
        this.getArcLayer();//构建OD弧段图层
        this.getHeatMapLayer();//构建热力图图层
        this.getGridLayer();//构建格网图层
        this.getSpeedLayer();//构建速度图层
        this.toParent();//将每天的轨迹数目统计结果反馈给父组件
    }
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
                this.speedLayerShow?this.state.speedLayer:null,]}>
          <StaticMap mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN} mapStyle={'mapbox://styles/2017302590157/cksbi52rm50pk17npkgfxiwni'}/>
          { this._renderTooltip() }
          </DeckGL>
          <div className={`moudle`}>
            GridLayer   <Switch defaultChecked onChange={this.changeGridLayerShow}/><br />
            GridLayer3D <Switch defaultChecked onChange={this.changeGridLayer3D}/><br />
            SpeedLayer   <Switch onChange={this.changeSpeedLayerShow}/><br />
            SpeedLayer3D <Switch onChange={this.changeSpeedLayer3D}/><br />
            HeatMapLayer<Switch  onChange={this.changeHeatMapLayerShow} /><br />
            ArcLayer    <Switch  onChange={this.changeArcLayerShow} /><br />
          </div><br/>
          <div className={`moudle`} style = {{'textAlign':'center'}}>
            GridWidth   <Slider style = {{width:'140px'}} max={500} min={50} step={50} defaultValue={100} onChange={(value) => this.changeGridWidth(value)}/>
          </div>
          </div>
        )
    }
}
export default DeckGLMap