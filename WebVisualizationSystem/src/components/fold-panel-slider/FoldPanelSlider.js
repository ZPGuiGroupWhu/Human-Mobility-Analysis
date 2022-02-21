import React, { useEffect, useMemo, useRef, useState } from 'react';
import './FoldPanelSlider.scss';
import _ from 'lodash';
import { use } from 'echarts';

export default function FoldPanelSlider(props) {
  const { style, mainComponents, minorComponents, setBottomDrawerHeight } = props;
  const footerHeight = '15px';

  const mains = useMemo(() => (Array.isArray(mainComponents) ? mainComponents : [mainComponents]), [mainComponents]);
  const minors = useMemo(() => (Array.isArray(minorComponents) ? minorComponents : [minorComponents]), [minorComponents]);

  const [isFold, setFold] = useState(true); // 是否折叠
  function handleClick(e) {
    setFold(prev => (!prev));
  }

  const handleMouseLeave = useMemo(() => { return _.debounce(() => { setFold(true) }, 300) }, []);


  return (
    <div className='fold-panel-slider-ctn' style={{ ...style }} onMouseLeave={handleMouseLeave}>
      <div style={{margin: footerHeight}}>
        <section className='fold-panel-slider-content'>
          {mains}
        </section>
        <section
          className='fold-panel-slider-content minor'
          style={{ overflow: 'hidden', maxHeight: isFold ? '0px' : '400px' }}
        >
          {minors}
        </section>
      </div>
      <footer style={{height: footerHeight}} className='fold-panel-slider-footer' onClick={handleClick}></footer>
    </div>
  )
}