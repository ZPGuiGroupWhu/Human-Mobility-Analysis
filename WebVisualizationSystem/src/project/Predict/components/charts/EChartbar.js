import React, { useEffect } from 'react';
import "./EChartbar.scss";

const EChartbar = (props) => {
  useEffect(()=>{
    const toLeft = {
      origin: 'right',
      distance: '100px',
    }
    window.ScrollReveal().reveal('.echart-bar-ctn', toLeft);
  }, [])

  return (
    <div className="echart-bar-ctn load-hidden">
      {props.children}
    </div>
  )
}

export default EChartbar;