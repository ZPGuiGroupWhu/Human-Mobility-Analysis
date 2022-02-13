import React, { useEffect, useState } from 'react';
import './FoldPanel.scss';

export default function FoldPanel(props) {
  const {
    width,
    renderEntryComponent,
    renderExpandComponent,
  } = props;

  const [isFold, setFold] = useState(true); // 是否处于折叠状态
  const [height, setHeight] = useState(0); // 容器高度


  useEffect(() => {
    if (isFold) {
      setHeight(document.querySelector('#fold-panel-entry').offsetHeight);
    } else {
      setHeight(document.querySelector('#fold-panel-expand').offsetHeight + document.querySelector('#fold-panel-entry').offsetHeight);
    }
  }, [isFold])

  return (
    <div id='fold-panel' style={{width, height}}>
      <section id='fold-panel-entry'>
        {renderEntryComponent(setFold)}
      </section>
      <section id='fold-panel-expand'>
        {renderExpandComponent()}
      </section>
    </div>
  )
}
