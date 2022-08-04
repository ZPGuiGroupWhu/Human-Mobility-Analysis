import React from 'react';
import "./EChartbar.scss";

const EChartbar = (props) => {

  return (
    <div className="echart-bar-ctn">
      {props.children}
    </div>
  )
}

export default EChartbar;