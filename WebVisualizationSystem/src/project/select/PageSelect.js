import React, { Component } from 'react';
import "./PageSelect.scss";
import Map from './map/Map';
import Footer from './footer/Footer';
import Charts from './charts/Charts';

class PageSelect extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  componentDidMount() {

  }

  componentDidUpdate(prevProps, prevState) {

  }

  componentWillUnmount() {

  }

  render() {
    return (
      <div className="select-page-ctn">
        <div className="center">
          <div className="inner">
            <div className="map">
              <Map />
            </div>
            <div className="footer-bar">
              <Footer />
            </div>
          </div>
        </div>
        <div className="left">
          <Charts.Group>
            <Charts.Box title="测试">
              <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
            </Charts.Box>
            <Charts.Box title="测试">
              <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
            </Charts.Box>
            <Charts.Box title="测试">
              <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
            </Charts.Box>
            <Charts.Box title="测试">
              <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
            </Charts.Box>
          </Charts.Group>
        </div>
        <div className="right">
          <Charts.Group>
          </Charts.Group>
        </div>
      </div>
    )
  }
}

export default PageSelect