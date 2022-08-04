import React, { useEffect } from 'react';
import "./Tooltip.scss";

export default function Tooltip(props) {
  const { top, left, display = 'none' } = props;
  const toTop = {
    distance: '50px',
    origin: 'bottom',
  }
  useEffect(() => {
    window.ScrollReveal().reveal('.tooltip-ctn', toTop);
  }, [])
  return (
    <div className="tooltip-ctn load-hidden" style={{ display, top, left }}>
      <div className="box">
        {props.children}
      </div>
      <div className="triangle"></div>
    </div>
  )
}