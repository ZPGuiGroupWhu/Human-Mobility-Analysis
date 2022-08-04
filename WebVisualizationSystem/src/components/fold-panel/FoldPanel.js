import React, { useEffect, useRef, useState } from 'react';
import './FoldPanel.scss';

export default function FoldPanel(props) {
  const {
    width,
    renderEntryComponent,
    renderExpandComponent,
    className = '',
    id,
  } = props;

  const [isFold, setFold] = useState(true); // 是否处于折叠状态
  const [height, setHeight] = useState(0); // 容器高度

  const ref = useRef(null);
  const entry = useRef(null);
  const expand = useRef(null);

  // 折叠效果
  useEffect(() => {
    if (isFold) {
      setHeight(entry.current.offsetHeight);
    } else {
      setHeight(entry.current.offsetHeight + expand.current.offsetHeight);
      // 每次高度发生变化时，都将组件滚动到合适的位置，展示全局
      // 300ms 后滚动，此时动画完成，可以正确获得高度
      setTimeout(() => {
        ref.current.scrollIntoView({behavior:'smooth', block: 'end'});
      }, 300)
    }
  }, [isFold])

  return (
    <div id={id} className={className.length ? `fold-panel ${className}` : 'fold-panel'} style={{ width, height }} ref={ref}>
      <section className='fold-panel-entry' ref={entry}>
        {renderEntryComponent(setFold)}
      </section>
      <section className='fold-panel-expand' ref={expand}>
        {renderExpandComponent()}
      </section>
    </div>
  )
}
