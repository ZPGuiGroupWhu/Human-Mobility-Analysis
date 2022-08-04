// 拖拽移动
import {useEffect} from 'react';

export const useDrag = function (dom) {
  useEffect(() => {
    if (!dom) return () => {};

    let isDraggable = false; // 是否可拖拽
    let mousePos = {x: 0, y: 0}; // 鼠标位置
    let targetPos = {x: 0, y: 0}; // 目标位置
  
    const onMouseDown = (e) => {
      mousePos = {
        x: e.clientX,
        y: e.clientY,
      };
      targetPos = {
        x: dom.offsetLeft,
        y: dom.offsetTop,
      }
      if (!isDraggable) {
        isDraggable = true;
      }
    }
    const onMouseMove = (e) => {
      if (!isDraggable) return;
      const curMousePos = {
        x: e.clientX,
        y: e.clientY,
      };
      const deltaMousePos = {
        deltaX: (curMousePos.x - mousePos.x),
        deltaY: (curMousePos.y - mousePos.y),
      };
      dom.style.setProperty('left', `${targetPos.x + deltaMousePos.deltaX}px`);
      dom.style.setProperty('top', `${targetPos.y + deltaMousePos.deltaY}px`);
    }
    const onMouseUp = (e) => {
      if (isDraggable) {
        isDraggable = false;
      }
    }
    
    dom.addEventListener('mousedown', onMouseDown, false);
    dom.addEventListener('mousemove', onMouseMove, false);
    dom.addEventListener('mouseup', onMouseUp, false);
    return () => {
      dom.removeEventListener('mousedown', onMouseDown);
      dom.removeEventListener('mousemove', onMouseMove);
      dom.removeEventListener('mouseup', onMouseUp);
    }
  }, [dom])
}