import React, { Component } from 'react';
import DeckGL from '@deck.gl/react';
import { StaticMap } from 'react-map-gl';
import { HeatmapLayer, GPUGridLayer } from '@deck.gl/aggregation-layers';
import { ArcLayer } from '@deck.gl/layers';
import './DeckGLMap.css'

const MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiMjAxNzMwMjU5MDE1NyIsImEiOiJja3FqM3RjYmIxcjdyMnhsbmR0bHo2ZGVpIn0.wNBmzyxhzCMx9PhIH3rwCA';//MAPBOX密钥

class DeckGLMap extends Component {
  constructor(props) {
    super(props);
    this.trajNodes = [];
    this.state = {
      arcLayer: null,
      heatMapLayer: null,
      gridLayer: null,
    }
  }
  componentDidMount() {
    this.getLayers();
  }
  componentDidUpdate(prevProps) {
    if (prevProps.userData !== this.props.userData) {
      this.getLayers();
    }
  }
  getTrajNodes = () => {
    let Nodes = []
    for (let i = 0; i < this.props.userData.length; i++) {
      for (let j = 0; j < this.props.userData[i].data.length; j++) {
        Nodes.push({ COORDINATES: this.props.userData[i].data[j], WEIGHT: 1 });
      }
    }
    this.trajNodes = Nodes;
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