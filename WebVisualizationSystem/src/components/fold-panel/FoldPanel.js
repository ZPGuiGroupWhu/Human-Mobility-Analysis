import React, { useEffect, useRef, useState } from 'react';
import './FoldPanel.scss';

export default function FoldPanel(props) {
  const {
    width,
    renderEntryComponent,
    renderExpandComponent,
  } = props;

  const [isFold, setFold] = useState(true); // 是否处于折叠状态
  const [height, setHeight] = useState(0); // 容器高度

  const entry = useRef(null);
  const expand = useRef(null);


  useEffect(() => {
    if (isFold) {
      setHeight(entry.current.offsetHeight);
    } else {
      setHeight(entry.current.offsetHeight + expand.current.offsetHeight);
    }
  }, [isFold])

  return (
    <div className='fold-panel' style={{width, height}}>
      <section className='fold-panel-entry' ref={entry}>
        {renderEntryComponent(setFold)}
      </section>
      <section className='fold-panel-expand' ref={expand}>
        {renderExpandComponent()}
      </section>
    </div>
  )
}
