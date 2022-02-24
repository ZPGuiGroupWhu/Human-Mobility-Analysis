import React, { useMemo, useState } from 'react';
import './FoldPanelSlider.scss';
import _ from 'lodash';
import { CaretUpOutlined, CaretDownOutlined } from '@ant-design/icons';

export default function FoldPanelSlider(props) {
  const { style, mainComponents, minorComponents } = props;
  const footerHeight = '15px';

  const mains = useMemo(() => (Array.isArray(mainComponents) ? mainComponents : [mainComponents]), [mainComponents]);
  const minors = useMemo(() => (Array.isArray(minorComponents) ? minorComponents : [minorComponents]), [minorComponents]);

  const [isFold, setFold] = useState(true); // 是否折叠
  function handleClick(e) {
    setFold(prev => (!prev));
  }

  const [isIconShow, setIconShow] = useState(false); // 是否显示icon提示

  const handleMouseLeave = useMemo(() => { return _.debounce(() => { setFold(true); setIconShow(false) }, 300) }, []);
  const handleMouseEnter = () => { 
    setIconShow(true); 
    handleMouseLeave.cancel(); // 取消防抖注册的延迟回调 
  };


  return (
    <div
      className='fold-panel-slider-ctn'
      style={{ ...style }}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
    >
      <div style={{ margin: footerHeight }}>
        <section className='fold-panel-slider-content'>
          {mains}
        </section>
        <section
          className='fold-panel-slider-content minor'
          style={{ maxHeight: isFold ? '0px' : '400px' }}
        >
          {minors}
        </section>
      </div>

      <footer style={{ height: footerHeight }} className='fold-panel-slider-footer' onClick={handleClick}>
        {
          isFold ? (
            <CaretUpOutlined
              className='icon-animation'
              style={{
                color: '#fff',
                position: 'absolute',
                bottom: !isIconShow ? '-30px' : '-4px',
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '30px'
              }}
            />
          ) : (
            <CaretDownOutlined
              className='icon-animation'
              style={{
                color: '#fff',
                position: 'absolute',
                bottom: !isIconShow ? '-30px' : '-4px',
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '30px'
              }}
            />
          )
        }
      </footer>
    </div>
  )
}