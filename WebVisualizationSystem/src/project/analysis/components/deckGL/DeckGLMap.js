import React, { Component } from 'react';
import DeckGL from '@deck.gl/react';
import { StaticMap } from 'react-map-gl';
import { HeatmapLayer, GPUGridLayer } from '@deck.gl/aggregation-layers';
import { ArcLayer } from '@deck.gl/layers';
import './DeckGLMap.css'
import { eventEmitter } from '@/common/func/EventEmitter';
import _ from 'lodash';

const MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiMjAxNzMwMjU5MDE1NyIsImEiOiJja3FqM3RjYmIxcjdyMnhsbmR0bHo2ZGVpIn0.wNBmzyxhzCMx9PhIH3rwCA';//MAPBOX密钥

class DeckGLMap extends Component {
  constructor(props) {
    super(props);
    this.trajNodes = [];
    this.state = {
      arcLayer: null,
      heatMapLayer: null,
      gridLayer: null,
      trajCounts: null,
      // selectDate: {
      //   start: '',
      //   end: '',
      // }
    }
  }
  componentDidMount() {
    this.getLayers();
    // this.addDateSelectListener();
  }
  componentDidUpdate(prevProps, prevState) {
    if (prevProps.userData !== this.props.userData) {
      this.getLayers();
    }
    // if (!_.isEqual(prevState.selectDate, this.state.selectDate)) {
    //   // do something here: {start: '...', end: '...'}
    //   console.log(this.state.selectDate);
    // }
  }
  getTrajNodes = () => {
    let Nodes = []//统计所有节点的坐标
    let Count = {}//统计每天的轨迹数目
    for (let i = 0; i < this.props.userData.length; i++) {
      if (Count[this.props.userData[i].date] === undefined) {
        Count[this.props.userData[i].date] = { 'count': 1 }
      }
      else {
        Count[this.props.userData[i].date] = { 'count': Count[this.props.userData[i].date].count + 1 }
      }
      for (let j = 0; j < this.props.userData[i].data.length; j++) {
        Nodes.push({ COORDINATES: this.props.userData[i].data[j], WEIGHT: 1 });
      }
    }
    this.trajNodes = Nodes;
    this.trajCounts = Count;
  }
  toParent = () => {//将每天的轨迹数目统计结果反馈给父组件
    this.props.getTrajCounts(this.trajCounts)
  }
  getArcLayer = () => {
    this.setState({
      arcLayer: new ArcLayer({
        id: 'arc-layer',
        data: this.props.userData,
        pickable: true,
        getWidth: 3,
        getSourcePosition: d => d.O,
        getTargetPosition: d => d.D,
        getSourceColor: [255, 250, 97],
        getTargetColor: [30, 20, 255],
      })
    });
  }
  getHeatMapLayer = () => {
    this.setState({
      heatMapLayer: new HeatmapLayer({
        id: 'heatmapLayer',
        data: this.trajNodes,
        getPosition: d => d.COORDINATES,
        getWeight: d => d.WEIGHT,
        aggregation: 'SUM'
      })
    })
  }
  getGridLayer = () => {
    this.setState({
      gridLayer: new GPUGridLayer({
        id: 'gpu-grid-layer',
        data: this.trajNodes,
        pickable: true,
        extruded: true,
        // elevationDomain:[100,10000],
        cellSize: 100,
        elevationScale: 4,
        getPosition: d => d.COORDINATES
      })
    })
  }
  getLayers = () => {
    this.getTrajNodes();
    this.getArcLayer();
    this.getHeatMapLayer();
    this.getGridLayer();
    this.toParent()
  }

  addDateSelectListener() {
    eventEmitter.on(this.props.eventName, ({ start, end }) => {
      this.setState({
        selectDate: {
          start,
          end,
        }
      })
    })
  }

  render() {
    return (
      <DeckGL
        initialViewState={{
          longitude: 114.17,
          latitude: 22.65,
          zoom: 10,
          pitch: 45,
          bearing: 0
        }}
        controller={true}
        layers={[this.state.heatMapLayer, this.state.gridLayer]}>
        <StaticMap mapboxApiAccessToken={MAPBOX_ACCESS_TOKEN} mapStyle={'mapbox://styles/mapbox/dark-v10'} />
      </DeckGL>
    )
  }
}
export default DeckGLMap