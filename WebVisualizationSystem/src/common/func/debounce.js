// 防抖
export const debounce = (fn, wait, immediate = false) => {
  let timer = null;

  function mainFunc (...args) {
    const ctx = this;

    // 函数执行前，必须清除定时器，目的是重置定时器
    if (timer) {
      clearTimeout(timer);
    };
    if (immediate) {
      // 保存当前的timer状态
      let callNow = !timer;
      // 重置定时器
      timer = setTimeout(() => {
        timer = null;
      }, wait)
      callNow && fn.call(ctx, ...args);
    } else {
      // 重置定时器
      timer = setTimeout(() => {
        fn.call(ctx, ...args);
        timer = null;
      }, wait)
    }
  }

  // 清除防抖功能
  mainFunc.cancel = function () {
    clearTimeout(timer);
    timer = null;
  }

  return mainFunc;
}