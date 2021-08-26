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
          <Charts></Charts>
        </div>
        <div className="right">
          <Charts></Charts>
        </div>
      </div>
    )
  }
}

export default PageSelect