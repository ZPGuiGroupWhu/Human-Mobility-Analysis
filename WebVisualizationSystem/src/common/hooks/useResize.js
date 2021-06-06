import { useState, useEffect } from 'react';
import { throttle } from '@/common/func/throttle.js'; // 节流

/**
 * 监听窗口大小变化：自适应调整页面布局
 * @param {number} time - 监听时间间隔
 * @returns {boolean} 交互状态
 */
function useResize(time) {
  const [isResize, setSize] = useState(false);

  const addListenerToResize = throttle(() => {
    setSize(true);
  }, time)

  // window.onresize 接受一个函数引用：
  // 当窗口大小变化时触发注册的函数。可结合节流函数使用，减少触发频率。
  window.onresize = addListenerToResize;

  useEffect(() => {
    // 重置交互状态
    isResize && setSize(false)
  }, [isResize])

  return isResize
}

export { useResize };